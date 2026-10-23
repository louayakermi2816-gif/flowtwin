"""Rule-selection pilot: once following a dispatching rule is a single action
(flowtwin.rule_actions), can the agents beat the planner's rules at w_tardiness = 2.5? Same
agents, budget, seeds, reward and evaluation as the sensitivity study; only the action space
changes. The protocol below was fixed before any of the pilot's runs and is written at the top
of the report.

Needs the study's runs at w_tardiness 2.5 in results/runs/ (scripts/run_sensitivity.py).

    python scripts/run_rule_pilot.py --workers 2      # train the missing runs, then write the report
    python scripts/run_rule_pilot.py --report-only    # rebuild results/rule_selection_pilot.md from results/runs/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import QNet, policy_from_qnet
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, paired_ci, paired_summary, policy_behaviour, summarize
from flowtwin.ppo import ActorCritic, policy_from_model
from flowtwin.rule_actions import RULES, RuleSelectionEnv, line_policy
from run_dense_pilot import LAST_EPISODES, training_return
from run_sensitivity import AGENTS, EPISODES, EVAL_SEED0, MODELS, RUNS, SEEDS, TRAIN_SEED0, run_name, study_weight

W = 2.5

PROTOCOL = f"""Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.rule_actions`, opt-in). Each decision picks a dispatching
  rule ({', '.join(RULES)}), which chooses the product, and a line speed: 9 actions, as before. Observation,
  reward and line are unchanged. Following EDD order is one repeated action, and every edd_slack decision
  is an available action.
- **Setting**: as in the sensitivity study at w_tardiness = {W:g}: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions, training seeds {', '.join(map(str, SEEDS))}; every other reward weight keeps its default.
- **Evaluation**: seeds {EVAL_SEED0}–{EVAL_SEED0 + EPISODES - 1} for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  {TRAIN_SEED0}–{TRAIN_SEED0 + EPISODES - 1}, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: rule selection lets an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed;
  for every agent its share of decisions per rule (rule-action agents only), speed shares, how often its
  launch order equals the EDD order, its most common launch order and the mean return of its last
  {LAST_EPISODES} training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions."""


def rules_name(seed: int) -> str:
    return "rules_" + run_name(W, seed)


def train_missing(steps: int, workers: int) -> None:
    jobs = [(agent, seed) for seed in SEEDS for agent in AGENTS
            if not (RUNS / rules_name(seed) / AGENTS[agent][1]).exists()]
    print(f"{len(jobs)} training runs to do", flush=True)

    def run(job):
        agent, seed = job
        out, models = RUNS / rules_name(seed), MODELS / rules_name(seed)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", AGENTS[agent][0], "--steps", str(steps), "--seed", str(seed),
               "--w-tardiness", str(W), "--rule-actions", "--out", str(out), "--models", str(models)]
        with open(out / f"{agent}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{agent} rules seed={seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(pool.map(run, jobs)):
            sys.exit("some training runs failed; see results/runs/rules_*/*.log")


def load_rules(agent: str, seed: int):
    """Saved evaluation and the greedy policy over (rule, speed) actions, unmasked like in training."""
    _, result, row, ckpt, _ = AGENTS[agent]
    data = json.loads((RUNS / rules_name(seed) / result).read_text(encoding="utf-8"))
    env = RuleSelectionEnv(FlowTwinEnv(DEFAULT))
    net = (ActorCritic if agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODELS / rules_name(seed) / ckpt, weights_only=True))
    net.eval()
    policy = policy_from_model(net, masked=False) if agent == "ppo" else policy_from_qnet(net, masked=False)
    return {c: np.array(v) for c, v in data[row].items()}, policy


def behaviour_with_rules(policy, cfg) -> dict:
    """policy_behaviour on the line actions, plus the share of decisions given to each rule."""
    picks = Counter()

    def counted(env, obs):
        action = policy(env, obs)
        picks[action // cfg.n_speeds] += 1
        return action

    b = policy_behaviour(line_policy(counted), cfg, EPISODES, EVAL_SEED0)
    b["rule_share"] = [picks[i] / sum(picks.values()) for i in range(len(RULES))]
    return b


def off(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} s{seed}"           # the study's row name


def on(agent: str, seed: int) -> str:
    return f"{AGENTS[agent][4]} rules s{seed}"


def pilot() -> dict:
    r = study_weight(W)            # rules, best rule and product-action agents, computed as in the study
    cfg = replace(DEFAULT, w_tardiness=W)
    r["rule_wins"], r["train_return"] = {}, {}
    for agent in AGENTS:
        r["rule_wins"][agent] = 0
        for seed in SEEDS:
            res, policy = load_rules(agent, seed)
            again = evaluate(line_policy(policy), cfg, EPISODES, EVAL_SEED0)["return"]
            if not np.array_equal(again, res["return"]):
                sys.exit(f"{on(agent, seed)}: the loaded policy does not reproduce its saved evaluation")
            r["results"][on(agent, seed)] = res
            mean, half = paired_ci(res["return"] - r["results"][r["best"]]["return"])
            r["rule_wins"][agent] += int(mean - half > 0)
            r["behaviour"][on(agent, seed)] = behaviour_with_rules(policy, cfg)
            r["train_return"][off(agent, seed)] = training_return(run_name(W, seed), agent)
            r["train_return"][on(agent, seed)] = training_return(rules_name(seed), agent)
    return r


def write_report(r: dict, path: Path) -> None:
    res, best = r["results"], r["best"]

    def by_seed(fmt):
        return ", ".join(fmt(s) for s in SEEDS)

    def mean_return(name):
        return "{:.1f}".format(res[name]["return"].mean())

    def delta(agent, s):
        return "{:+.1f} ± {:.1f}".format(*paired_ci(res[on(agent, s)]["return"] - res[off(agent, s)]["return"]))

    b = res[best]["return"]
    lines = [f"# Rule-selection pilot (w_tardiness = {W:g})", "", PROTOCOL, "", "## Summary", "",
             f"Best rule, re-selected on training seeds: **{best}**, evaluation return {b.mean():.1f} ± {b.std():.1f}.", "",
             "| agent | eval return by seed, product actions | eval return by seed, rule actions "
             "| rule − product actions, paired Δ return by seed | launch order = EDD by seed, product → rule actions "
             "| rule actions beat the rules |",
             "|---|---|---|---|---|---|"]
    for agent in AGENTS:
        k = r["rule_wins"][agent]
        lines.append(
            f"| {AGENTS[agent][4]} | {by_seed(lambda s: mean_return(off(agent, s)))} "
            f"| {by_seed(lambda s: mean_return(on(agent, s)))} | {by_seed(lambda s: delta(agent, s))} "
            f"| {by_seed(lambda s: str(r['behaviour'][off(agent, s)]['edd_order']))} → "
            f"{by_seed(lambda s: str(r['behaviour'][on(agent, s)]['edd_order']))} (of {EPISODES}) "
            f"| {'yes' if k >= 2 else 'no'} ({k}/{len(SEEDS)} seeds) |")

    lines += ["", "## Details", "", "Evaluation seeds, mean ± std:", "", summarize(res), "",
              "Paired against the best rule:", "", paired_summary(res, best), "",
              "Rule actions against the product-action agent with the same training seed:", "",
              "| rule vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |", "|---|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            pair = {off(agent, s): res[off(agent, s)], on(agent, s): res[on(agent, s)]}
            lines.append(paired_summary(pair, off(agent, s)).splitlines()[-1])
    lines += ["", f"| agent | rule share {' / '.join(RULES)} | speed share low / nominal / high | launch order = EDD order "
              f"| most common launch order | training return, last {LAST_EPISODES} episodes |", "|---|---|---|---|---|---|"]
    for agent in AGENTS:
        for s in SEEDS:
            for name in (off(agent, s), on(agent, s)):
                bh = r["behaviour"][name]
                rules = " / ".join(f"{100 * x:.1f}%" for x in bh["rule_share"]) if "rule_share" in bh else "–"
                share = " / ".join(f"{100 * x:.1f}%" for x in bh["speed_share"])
                seq, count = bh["sequences"].most_common(1)[0]
                lines.append(f"| {name} | {rules} | {share} | {bh['edd_order']}/{bh['episodes']} "
                             f"| {' → '.join(map(str, seq))} ({count}/{bh['episodes']}) | {r['train_return'][name]:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="skip training; use the runs already in results/runs/")
    ap.add_argument("--out", default="results/rule_selection_pilot.md")
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
