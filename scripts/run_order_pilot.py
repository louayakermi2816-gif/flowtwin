"""Order-first pilot: when the launch order is one decision at the start of each shift
(flowtwin.order_actions), do the agents discover a good order, and can they beat the planner's
rules at w_tardiness = 2.5? Same agents, budget, seeds, reward and evaluation as the sensitivity
study; only the action space changes. The protocol below was fixed before any of the pilot's runs
and is written at the top of the report.

Needs the study's runs at w_tardiness 2.5 in results/runs/ (scripts/run_sensitivity.py).

    python scripts/run_order_pilot.py --workers 2      # train the missing runs, then write the report
    python scripts/run_order_pilot.py --report-only    # rebuild results/order_first_pilot.md from results/runs/
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
from flowtwin.dqn import QNet
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, paired_ci, paired_summary, policy_behaviour, summarize
from flowtwin.heuristics import make_edd_slack
from flowtwin.order_actions import OrderFirstEnv, order_line_policy
from flowtwin.ppo import ActorCritic
from run_dense_pilot import LAST_EPISODES, training_return
from run_sensitivity import AGENTS, EPISODES, EVAL_SEED0, MODELS, RUNS, SEEDS, TRAIN_SEED0, run_name, study_weight

W = 2.5

PROTOCOL = f"""Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.order_actions`, opt-in). The first decision of a shift picks one
  of the six launch orders (nothing is launched, reward 0); every later decision picks the line speed of a batch
  of the next product in that order that still has demand. The observation adds a planning flag and the chosen
  order; reward and line are unchanged. Planning the EDD order with the slack rule's speeds reproduces edd_slack
  exactly, so the best rule is one of the policies an agent can learn.
- **Setting**: as in the sensitivity study at w_tardiness = {W:g}: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions (planning decisions included), training seeds {', '.join(map(str, SEEDS))}; every other
  reward weight keeps its default.
- **Evaluation**: seeds {EVAL_SEED0}–{EVAL_SEED0 + EPISODES - 1} for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  {TRAIN_SEED0}–{TRAIN_SEED0 + EPISODES - 1}, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: order-first actions let an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed; for
  every agent how often its launch order equals the EDD order and its most common launch order, its speed shares,
  how often its action equals the tuned edd_slack rule's action in the same state, and the mean return of its
  last {LAST_EPISODES} training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions."""


def order_name(seed: int) -> str:
    return "order_" + run_name(W, seed)


def train_missing(steps: int, workers: int) -> None:
    jobs = [(agent, seed) for seed in SEEDS for agent in AGENTS
            if not (RUNS / order_name(seed) / AGENTS[agent][1]).exists()]
    print(f"{len(jobs)} training runs to do", flush=True)

    def run(job):
        agent, seed = job
        out, models = RUNS / order_name(seed), MODELS / order_name(seed)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", AGENTS[agent][0], "--steps", str(steps), "--seed", str(seed),
               "--w-tardiness", str(W), "--order-first", "--out", str(out), "--models", str(models)]
        with open(out / f"{agent}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{agent} order-first seed={seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(pool.map(run, jobs)):
            sys.exit("some training runs failed; see results/runs/order_*/*.log")


def load_order(agent: str, seed: int):
    """Saved evaluation and the greedy order-first policy on the plain env, as in training."""
    _, result, row, ckpt, _ = AGENTS[agent]
    data = json.loads((RUNS / order_name(seed) / result).read_text(encoding="utf-8"))
    env = OrderFirstEnv(FlowTwinEnv(DEFAULT))
    net = (ActorCritic if agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODELS / order_name(seed) / ckpt, weights_only=True))
    net.eval()
    policy = order_line_policy(lambda o, m: net.act(o, deterministic=True, mask=m))
    return {c: np.array(v) for c, v in data[row].items()}, policy


def behaviour_with_rule(policy, cfg, margin: float) -> dict:
    """policy_behaviour plus how often the policy takes the tuned slack rule's action in the same state."""
    rule, same = make_edd_slack(margin), [0, 0]

    def counted(env, obs):
        action = policy(env, obs)
        same[0] += int(action == rule(env, obs))
        same[1] += 1
        return action

    b = policy_behaviour(counted, cfg, EPISODES, EVAL_SEED0)
    b["rule_agreement"] = same[0] / same[1]
    return b


def off(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} s{seed}"           # the study's row name


def on(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} order-first s{seed}"


def pilot() -> dict:
    r = study_weight(W)            # rules, best rule and product-action agents, computed as in the study
    cfg = replace(DEFAULT, w_tardiness=W)
    r["order_wins"], r["train_return"] = {}, {}
    for agent in AGENTS:
        r["order_wins"][agent] = 0
        for seed in SEEDS:
            res, policy = load_order(agent, seed)
            again = evaluate(policy, cfg, EPISODES, EVAL_SEED0)["return"]
            if not np.array_equal(again, res["return"]):
                sys.exit(f"{on(agent, seed)}: the loaded policy does not reproduce its saved evaluation")
            r["results"][on(agent, seed)] = res
            mean, half = paired_ci(res["return"] - r["results"][r["best"]]["return"])
            r["order_wins"][agent] += int(mean - half > 0)
            r["behaviour"][on(agent, seed)] = behaviour_with_rule(policy, cfg, r["margin"])
            r["train_return"][off(agent, seed)] = training_return(run_name(W, seed), agent)
            r["train_return"][on(agent, seed)] = training_return(order_name(seed), agent)
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
    lines = [f"# Order-first pilot (w_tardiness = {W:g})", "", PROTOCOL, "", "## Summary", "",
             f"Best rule, re-selected on training seeds: **{best}**, evaluation return {b.mean():.1f} ± {b.std():.1f}.", "",
             "| agent | eval return by seed, product actions | eval return by seed, order first "
             "| order first − best rule, paired Δ return by seed | launch order = EDD by seed, product actions → order first "
             "| order first beats the rules |",
             "|---|---|---|---|---|---|"]
    for agent in AGENTS:
        k = r["order_wins"][agent]
        lines.append(
            f"| {AGENTS[agent][4]} | {by_seed(lambda s: mean_return(off(agent, s)))} "
            f"| {by_seed(lambda s: mean_return(on(agent, s)))} "
            f"| {by_seed(lambda s: delta(on(agent, s), best))} "
            f"| {by_seed(lambda s: str(r['behaviour'][off(agent, s)]['edd_order']))} → "
            f"{by_seed(lambda s: str(r['behaviour'][on(agent, s)]['edd_order']))} (of {EPISODES}) "
            f"| {'yes' if k >= 2 else 'no'} ({k}/{len(SEEDS)} seeds) |")

    lines += ["", "## Details", "", "Evaluation seeds, mean ± std:", "", summarize(res), "",
              "Paired against the best rule:", "", paired_summary(res, best), "",
              "Order first against the product-action agent with the same training seed:", "",
              "| order first vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |", "|---|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            pair = {off(agent, s): res[off(agent, s)], on(agent, s): res[on(agent, s)]}
            lines.append(paired_summary(pair, off(agent, s)).splitlines()[-1])
    lines += ["", f"| agent | speed share low / nominal / high | launch order = EDD order | most common launch order "
              f"| action = edd_slack({r['margin']:g})'s | training return, last {LAST_EPISODES} episodes |", "|---|---|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            for name in (off(agent, s), on(agent, s)):
                bh = r["behaviour"][name]
                share = " / ".join(f"{100 * x:.1f}%" for x in bh["speed_share"])
                seq, count = bh["sequences"].most_common(1)[0]
                agree = f"{100 * bh['rule_agreement']:.1f}%" if "rule_agreement" in bh else "–"
                lines.append(f"| {name} | {share} | {bh['edd_order']}/{bh['episodes']} "
                             f"| {' → '.join(map(str, seq))} ({count}/{bh['episodes']}) | {agree} | {r['train_return'][name]:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="skip training; use the runs already in results/runs/")
    ap.add_argument("--out", default="results/order_first_pilot.md")
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
