"""Lateness-weight sensitivity study: can a learned policy beat the planner's rules once
lateness carries real weight? The protocol below was fixed before any of the study's runs and
is written at the top of the report.

    python scripts/run_sensitivity.py --workers 6      # train the missing runs, then write the report
    python scripts/run_sensitivity.py --report-only    # rebuild results/sensitivity.md from results/runs/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import QNet, policy_from_qnet
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, paired_ci, paired_summary, policy_behaviour, summarize
from flowtwin.heuristics import HEURISTICS, TABLE_RULES, make_edd_slack
from flowtwin.ppo import ActorCritic, policy_from_model

WEIGHTS = (0.10, 1.0, 2.5)
SEEDS = (0, 1, 2)
MARGINS = (0, 30, 60, 120, 180, 240, 360, 480, 720)
TRAIN_SEED0, EVAL_SEED0, EPISODES = 0, 1000, 30
RUNS, MODELS = Path("results/runs"), Path("models/runs")
AGENTS = {  # key: (training script, results file, row in that file, checkpoint, label)
    "ppo": ("scripts/train_ppo.py", "ppo_vs_heuristics.json", "ppo (ours)", "ppo.pt", "PPO (ours)"),
    "dqn": ("scripts/train_dqn.py", "dqn_vs_heuristics.json", "double-dqn", "dqn.pt", "Double-DQN"),
}

PROTOCOL = f"""Protocol, fixed before any of the study's runs:

- **Weights**: w_tardiness in {{{', '.join(f'{w:g}' for w in WEIGHTS)}}}; every other reward weight keeps its default.
- **Rules**: {', '.join(TABLE_RULES)} and edd_slack. The edd_slack margin is tuned per weight on training
  seeds {TRAIN_SEED0}–{TRAIN_SEED0 + EPISODES - 1} over {{{', '.join(f'{m:g}' for m in MARGINS)}}} minutes. The best rule per weight is the
  one with the highest mean return on those training seeds.
- **Agents**: PPO (ours) and Double-DQN with their default hyperparameters, 100k decisions, training
  seeds {', '.join(map(str, SEEDS))}.
- **Evaluation**: seeds {EVAL_SEED0}–{EVAL_SEED0 + EPISODES - 1} for every policy (same orders, same breakdowns); mean ± std
  and paired differences against the best rule with 95% Student-t confidence intervals.
- **Decision rule**: an agent beats the rules at a weight if its paired Δ return against the best
  rule has a 95% CI entirely above 0 for at least 2 of its 3 training seeds."""


def run_name(w: float, seed: int) -> str:
    return f"wt{w:g}_s{seed}"


def train_missing(steps: int, workers: int) -> None:
    jobs = [(agent, w, seed) for w in WEIGHTS for seed in SEEDS for agent in AGENTS
            if not (RUNS / run_name(w, seed) / AGENTS[agent][1]).exists()]
    print(f"{len(jobs)} training runs to do", flush=True)

    def run(job):
        agent, w, seed = job
        out, models = RUNS / run_name(w, seed), MODELS / run_name(w, seed)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", AGENTS[agent][0], "--steps", str(steps), "--seed", str(seed),
               "--w-tardiness", str(w), "--out", str(out), "--models", str(models)]
        with open(out / f"{agent}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{agent} w={w:g} seed={seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(pool.map(run, jobs)):
            sys.exit("some training runs failed; see results/runs/*/*.log")


def load_agent(agent: str, w: float, seed: int):
    _, result, row, ckpt, _ = AGENTS[agent]
    data = json.loads((RUNS / run_name(w, seed) / result).read_text(encoding="utf-8"))
    env = FlowTwinEnv(DEFAULT)
    net = (ActorCritic if agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODELS / run_name(w, seed) / ckpt, weights_only=True))
    net.eval()
    policy = policy_from_model(net) if agent == "ppo" else policy_from_qnet(net)
    return {c: np.array(v) for c, v in data[row].items()}, policy


def study_weight(w: float) -> dict:
    cfg = replace(DEFAULT, w_tardiness=w)
    tuned = {m: evaluate(make_edd_slack(m), cfg, EPISODES, TRAIN_SEED0)["return"].mean() for m in MARGINS}
    margin = max(MARGINS, key=lambda m: tuned[m])
    rules = {name: HEURISTICS[name] for name in TABLE_RULES}
    slack_name = f"edd_slack({margin:g})"
    rules[slack_name] = make_edd_slack(margin)
    train_means = {name: evaluate(pol, cfg, EPISODES, TRAIN_SEED0)["return"].mean()
                   for name, pol in rules.items() if name != slack_name}
    train_means[slack_name] = tuned[margin]
    best = max(train_means, key=lambda name: train_means[name])

    results = {name: evaluate(pol, cfg, EPISODES, EVAL_SEED0) for name, pol in rules.items()}
    wins, behaviour = {}, {}
    for agent in AGENTS:
        wins[agent] = 0
        for seed in SEEDS:
            res, policy = load_agent(agent, w, seed)
            name = f"{AGENTS[agent][4]} s{seed}"
            results[name] = res
            mean, half = paired_ci(res["return"] - results[best]["return"])
            wins[agent] += int(mean - half > 0)
            behaviour[name] = policy_behaviour(policy, cfg, EPISODES, EVAL_SEED0)
    return {"margin": margin, "tuned": tuned, "train_means": train_means, "best": best,
            "results": results, "wins": wins, "behaviour": behaviour}


def write_report(rows: dict, path: Path) -> None:
    def seed_range(r, agent):
        means = [r["results"][f"{AGENTS[agent][4]} s{s}"]["return"].mean() for s in SEEDS]
        return f"{min(means):.1f} to {max(means):.1f}"

    def verdict(r, agent):
        k = r["wins"][agent]
        return f"{'yes' if k >= 2 else 'no'} ({k}/{len(SEEDS)} seeds)"

    lines = ["# Lateness-weight sensitivity study", "", PROTOCOL, "", "## Summary", "",
             "| w_tardiness | best rule, eval return | PPO (ours), seed means | Double-DQN, seed means | PPO beats the rules | Double-DQN beats the rules |",
             "|---|---|---|---|---|---|"]
    for w, r in rows.items():
        b = r["results"][r["best"]]["return"]
        lines.append(f"| {w:g} | {r['best']}: {b.mean():.1f} ± {b.std():.1f} | {seed_range(r, 'ppo')} | {seed_range(r, 'dqn')} "
                     f"| {verdict(r, 'ppo')} | {verdict(r, 'dqn')} |")
    for w, r in rows.items():
        tuned = ", ".join(f"{m:g}: {v:.1f}" for m, v in r["tuned"].items())
        train = ", ".join(f"{n}: {v:.1f}" for n, v in r["train_means"].items())
        lines += ["", f"## w_tardiness = {w:g}", "",
                  f"edd_slack margin tuned on training seeds: **{r['margin']:g} min** (mean training return by margin: {tuned}).",
                  f"Best rule on training seeds: **{r['best']}** (mean training return: {train}).", "",
                  "Evaluation seeds, mean ± std:", "", summarize(r["results"]), "",
                  "Paired against the best rule:", "", paired_summary(r["results"], r["best"]), "",
                  "| agent | speed share low / nominal / high | launch order = EDD order |", "|---|---|---|"]
        for name, b in r["behaviour"].items():
            share = " / ".join(f"{100 * x:.1f}%" for x in b["speed_share"])
            lines.append(f"| {name} | {share} | {b['edd_order']}/{b['episodes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true", help="skip training; use the runs already in results/runs/")
    ap.add_argument("--out", default="results/sensitivity.md")
    args = ap.parse_args()
    if not args.report_only:
        train_missing(args.steps, args.workers)
    rows = {w: study_weight(w) for w in WEIGHTS}
    write_report(rows, Path(args.out))
    print(Path(args.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
