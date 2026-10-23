"""Speed-only pilot: with the EDD order fixed (flowtwin.rule_actions.SPEED_ONLY), can the agents time
the line speed better than the planner's rules at w_tardiness = 2.5? Same agents, budget, seeds,
reward and evaluation as the sensitivity study; only the action space changes. The protocol below
was fixed before any of the pilot's runs and is written at the top of the report.

Needs the study's runs at w_tardiness 2.5 in results/runs/ (scripts/run_sensitivity.py).

    python scripts/run_speed_pilot.py --workers 2      # train the missing runs, then write the report
    python scripts/run_speed_pilot.py --report-only    # rebuild results/speed_only_pilot.md from results/runs/
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
from flowtwin.heuristics import make_edd_slack
from flowtwin.ppo import ActorCritic, policy_from_model
from flowtwin.rule_actions import SPEED_ONLY, RuleSelectionEnv, line_policy
from run_dense_pilot import LAST_EPISODES, training_return
from run_sensitivity import AGENTS, EPISODES, EVAL_SEED0, MODELS, RUNS, SEEDS, TRAIN_SEED0, run_name, study_weight

W = 2.5

PROTOCOL = f"""Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.rule_actions.SPEED_ONLY`, opt-in). EDD chooses the product and
  each decision only picks the line speed (low, nominal or high): 3 actions. Observation, reward and line are
  unchanged. Every policy in this space launches in EDD order, and every EDD-order rule, the slack-aware one
  included, is one of them.
- **Setting**: as in the sensitivity study at w_tardiness = {W:g}: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions, training seeds {', '.join(map(str, SEEDS))}; every other reward weight keeps its default.
- **Evaluation**: seeds {EVAL_SEED0}–{EVAL_SEED0 + EPISODES - 1} for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  {TRAIN_SEED0}–{TRAIN_SEED0 + EPISODES - 1}, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: speed-only actions let an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed; for
  every speed-only agent its speed shares, how often its speed equals the tuned edd_slack rule's choice in the
  same state, and the mean return of its last {LAST_EPISODES} training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions."""


def speed_name(seed: int) -> str:
    return "speed_" + run_name(W, seed)


def train_missing(steps: int, workers: int) -> None:
    jobs = [(agent, seed) for seed in SEEDS for agent in AGENTS
            if not (RUNS / speed_name(seed) / AGENTS[agent][1]).exists()]
    print(f"{len(jobs)} training runs to do", flush=True)

    def run(job):
        agent, seed = job
        out, models = RUNS / speed_name(seed), MODELS / speed_name(seed)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", AGENTS[agent][0], "--steps", str(steps), "--seed", str(seed),
               "--w-tardiness", str(W), "--speed-only", "--out", str(out), "--models", str(models)]
        with open(out / f"{agent}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{agent} speed seed={seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(pool.map(run, jobs)):
            sys.exit("some training runs failed; see results/runs/speed_*/*.log")


def load_speed(agent: str, seed: int):
    """Saved evaluation and the greedy policy over speeds, unmasked like in training."""
    _, result, row, ckpt, _ = AGENTS[agent]
    data = json.loads((RUNS / speed_name(seed) / result).read_text(encoding="utf-8"))
    env = RuleSelectionEnv(FlowTwinEnv(DEFAULT), SPEED_ONLY)
    net = (ActorCritic if agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODELS / speed_name(seed) / ckpt, weights_only=True))
    net.eval()
    policy = policy_from_model(net, masked=False) if agent == "ppo" else policy_from_qnet(net, masked=False)
    return {c: np.array(v) for c, v in data[row].items()}, policy


def speed_behaviour(policy, cfg, margin: float) -> dict:
    """policy_behaviour on the line actions, plus how often the agent picks the speed that the
    tuned slack rule would pick in the same state."""
    rule, same = make_edd_slack(margin), [0, 0]

    def counted(env, obs):
        speed = policy(env, obs)
        same[0] += int(speed == env.decode(rule(env, obs))[1])
        same[1] += 1
        return speed

    b = policy_behaviour(line_policy(counted, SPEED_ONLY), cfg, EPISODES, EVAL_SEED0)
    b["slack_agreement"] = same[0] / same[1]
    return b


def off(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} s{seed}"           # the study's row name


def on(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} speed s{seed}"


def pilot() -> dict:
    r = study_weight(W)            # rules, best rule and product-action agents, computed as in the study
    cfg = replace(DEFAULT, w_tardiness=W)
    r["speed_wins"], r["train_return"] = {}, {}
    for agent in AGENTS:
        r["speed_wins"][agent] = 0
        for seed in SEEDS:
            res, policy = load_speed(agent, seed)
            again = evaluate(line_policy(policy, SPEED_ONLY), cfg, EPISODES, EVAL_SEED0)["return"]
            if not np.array_equal(again, res["return"]):
                sys.exit(f"{on(agent, seed)}: the loaded policy does not reproduce its saved evaluation")
            r["results"][on(agent, seed)] = res
            mean, half = paired_ci(res["return"] - r["results"][r["best"]]["return"])
            r["speed_wins"][agent] += int(mean - half > 0)
            r["behaviour"][on(agent, seed)] = speed_behaviour(policy, cfg, r["margin"])
            r["train_return"][off(agent, seed)] = training_return(run_name(W, seed), agent)
            r["train_return"][on(agent, seed)] = training_return(speed_name(seed), agent)
    return r


def write_report(r: dict, path: Path) -> None:
    res, best = r["results"], r["best"]

    def by_seed(fmt):
        return ", ".join(fmt(s) for s in SEEDS)

    def mean_return(name):
        return "{:.1f}".format(res[name]["return"].mean())

    def delta(name, ref):
        return "{:+.1f} ± {:.1f}".format(*paired_ci(res[name]["return"] - res[ref]["return"]))

    b = res[best]["return"]
    lines = [f"# Speed-only pilot (w_tardiness = {W:g})", "", PROTOCOL, "", "## Summary", "",
             f"Best rule, re-selected on training seeds: **{best}**, evaluation return {b.mean():.1f} ± {b.std():.1f}.", "",
             "| agent | eval return by seed, product actions | eval return by seed, speed only "
             "| speed only − product actions, paired Δ return by seed | speed only − best rule, paired Δ return by seed "
             "| speed only beats the rules |",
             "|---|---|---|---|---|---|"]
    for agent in AGENTS:
        k = r["speed_wins"][agent]
        lines.append(
            f"| {AGENTS[agent][4]} | {by_seed(lambda s: mean_return(off(agent, s)))} "
            f"| {by_seed(lambda s: mean_return(on(agent, s)))} "
            f"| {by_seed(lambda s: delta(on(agent, s), off(agent, s)))} "
            f"| {by_seed(lambda s: delta(on(agent, s), best))} "
            f"| {'yes' if k >= 2 else 'no'} ({k}/{len(SEEDS)} seeds) |")

    lines += ["", "## Details", "", "Evaluation seeds, mean ± std:", "", summarize(res), "",
              "Paired against the best rule:", "", paired_summary(res, best), "",
              "Speed only against the product-action agent with the same training seed:", "",
              "| speed only vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |", "|---|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            pair = {off(agent, s): res[off(agent, s)], on(agent, s): res[on(agent, s)]}
            lines.append(paired_summary(pair, off(agent, s)).splitlines()[-1])
    lines += ["", f"| agent | speed share low / nominal / high | speed = edd_slack({r['margin']:g})'s choice "
              f"| training return, last {LAST_EPISODES} episodes |", "|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            for name in (off(agent, s), on(agent, s)):
                bh = r["behaviour"][name]
                share = " / ".join(f"{100 * x:.1f}%" for x in bh["speed_share"])
                agree = f"{100 * bh['slack_agreement']:.1f}%" if "slack_agreement" in bh else "–"
                lines.append(f"| {name} | {share} | {agree} | {r['train_return'][name]:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="skip training; use the runs already in results/runs/")
    ap.add_argument("--out", default="results/speed_only_pilot.md")
    args = ap.parse_args()
    missing = [run_name(W, s) for s in SEEDS if not (RUNS / run_name(W, s)).exists()]
    if missing:
        sys.exit(f"missing study runs {missing}: run scripts/run_sensitivity.py first")
    if not args.report_only:
        train_missing(args.steps, args.workers)
    write_report(pilot(), Path(args.out))
    print(Path(args.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
