"""Warm-start pilot: if PPO starts from the best rule's own decisions (flowtwin.imitation), does RL
keep, improve or lose the rule at w_tardiness = 2.5? Same agent, budget, seeds, reward and
evaluation as the sensitivity study; only the starting weights change. The protocol below was
fixed before any of the pilot's runs and is written at the top of the report.

Needs the study's runs at w_tardiness 2.5 in results/runs/ (scripts/run_sensitivity.py).

    python scripts/run_warm_start_pilot.py --workers 2      # train the missing runs, then write the report
    python scripts/run_warm_start_pilot.py --report-only    # rebuild results/warm_start_pilot.md from results/runs/
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
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, paired_ci, paired_summary, policy_behaviour, summarize
from flowtwin.heuristics import make_edd_slack
from flowtwin.imitation import CloneConfig
from flowtwin.ppo import ActorCritic, policy_from_model
from run_dense_pilot import LAST_EPISODES, training_return
from run_sensitivity import (EPISODES, EVAL_SEED0, MARGINS, MODELS, RUNS, SEEDS, TRAIN_SEED0, load_agent, run_name,
                             study_weight)

W = 2.5
CLONE = CloneConfig()
STUDY, BEFORE, AFTER = "PPO (ours) s{}", "PPO warm start, before RL s{}", "PPO warm start + RL s{}"

PROTOCOL = f"""Protocol, fixed before any of the pilot's runs:

- **Change under test**: a warm start (`flowtwin.imitation`, opt-in). Before RL, PPO's actor-critic is fitted to the
  decisions of edd_slack, with the margin tuned exactly as in the study, on {CLONE.episodes} demonstration shifts (seeds
  {CLONE.seed0}–{CLONE.seed0 + CLONE.episodes - 1}): the actor to the rule's actions (cross-entropy, invalid actions masked) and the critic
  to the rule's discounted return at PPO's reward scale; {CLONE.epochs} epochs, minibatch {CLONE.minibatch}, Adam learning rate {CLONE.lr:g}.
  The {CLONE.epochs} epochs were chosen before any pilot run, on held-out demonstration shifts 200–249 (training seed 0:
  held-out accuracy 0.917, 0.957 and 0.969 after 20, 50 and 100 epochs). PPO then trains for 100k decisions from
  those weights with its default hyperparameters; action space, observation, reward and line are unchanged.
- **Agent**: PPO (ours) only, training seeds {', '.join(map(str, SEEDS))}. Double-DQN is left out: its ε-greedy schedule
  starts fully random, so a warm start would also need a schedule change, which is a hyperparameter change.
- **Evaluation**: seeds {EVAL_SEED0}–{EVAL_SEED0 + EPISODES - 1} for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study; the study's PPO with the same seed (no warm
  start); and the warm-started policy before RL.
- **Decision rule**: the warm start lets PPO beat the rules if the trained agent's paired Δ return against the best
  rule has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ of the trained agent against the policy before RL and
  against the study's PPO; for every policy its speed shares, how often its launch order equals the EDD order, how
  often its action equals the teacher's action in the same state and its most common launch order; the mean
  return of the last {LAST_EPISODES} training episodes; the clone's accuracy on its demonstrations.
- **Scope**: a pilot. It changes no published result."""


def warm_name(seed: int) -> str:
    return "warm_" + run_name(W, seed)


def tuned_margin() -> float:
    """The edd_slack margin exactly as study_weight tunes it: the highest mean return on the training seeds."""
    cfg = replace(DEFAULT, w_tardiness=W)
    tuned = {m: evaluate(make_edd_slack(m), cfg, EPISODES, TRAIN_SEED0)["return"].mean() for m in MARGINS}
    return max(MARGINS, key=lambda m: tuned[m])


def train_missing(steps: int, workers: int, margin: float) -> None:
    jobs = [seed for seed in SEEDS if not (RUNS / warm_name(seed) / "ppo_vs_heuristics.json").exists()]
    print(f"{len(jobs)} training runs to do, teacher edd_slack({margin:g})", flush=True)

    def run(seed):
        out, models = RUNS / warm_name(seed), MODELS / warm_name(seed)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/train_ppo.py", "--steps", str(steps), "--seed", str(seed),
               "--w-tardiness", str(W), "--warm-start-slack", str(margin), "--out", str(out), "--models", str(models)]
        with open(out / "ppo.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"ppo warm start seed={seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(pool.map(run, jobs)):
            sys.exit("some training runs failed; see results/runs/warm_*/ppo.log")


def load(seed: int, ckpt: str, row: str):
    """Saved evaluation and greedy policy of one of a warm-start run's two checkpoints."""
    data = json.loads((RUNS / warm_name(seed) / "ppo_vs_heuristics.json").read_text(encoding="utf-8"))
    env = FlowTwinEnv(DEFAULT)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODELS / warm_name(seed) / ckpt, weights_only=True))
    net.eval()
    return {c: np.array(v) for c, v in data[row].items()}, policy_from_model(net)


def behaviour_with_teacher(policy, cfg, margin: float) -> dict:
    """policy_behaviour plus how often the policy takes the teacher's action in the same state."""
    teacher, same = make_edd_slack(margin), [0, 0]

    def counted(env, obs):
        action = policy(env, obs)
        same[0] += int(action == teacher(env, obs))
        same[1] += 1
        return action

    b = policy_behaviour(counted, cfg, EPISODES, EVAL_SEED0)
    b["teacher_agreement"] = same[0] / same[1]
    return b


def pilot(margin: float) -> dict:
    r = study_weight(W)            # rules, best rule and the study's agents, computed as in the study
    if r["margin"] != margin:
        sys.exit(f"the study's tuned margin {r['margin']:g} differs from the teacher's {margin:g}")
    cfg = replace(DEFAULT, w_tardiness=W)
    r["results"] = {k: v for k, v in r["results"].items() if not k.startswith("Double-DQN")}
    r["behaviour"], r["train_return"], r["clone"], r["warm_wins"] = {}, {}, {}, 0
    for seed in SEEDS:
        _, policy = load_agent("ppo", W, seed)
        r["behaviour"][STUDY.format(seed)] = behaviour_with_teacher(policy, cfg, margin)
        for name, ckpt, row in ((BEFORE.format(seed), "ppo_warm_start.pt", "ppo warm start (before rl)"),
                                (AFTER.format(seed), "ppo.pt", "ppo (ours)")):
            res, policy = load(seed, ckpt, row)
            again = evaluate(policy, cfg, EPISODES, EVAL_SEED0)["return"]
            if not np.array_equal(again, res["return"]):
                sys.exit(f"{name}: the loaded policy does not reproduce its saved evaluation")
            r["results"][name] = res
            r["behaviour"][name] = behaviour_with_teacher(policy, cfg, margin)
        mean, half = paired_ci(r["results"][AFTER.format(seed)]["return"] - r["results"][r["best"]]["return"])
        r["warm_wins"] += int(mean - half > 0)
        r["train_return"][STUDY.format(seed)] = training_return(run_name(W, seed), "ppo")
        r["train_return"][AFTER.format(seed)] = training_return(warm_name(seed), "ppo")
        r["clone"][seed] = json.loads((RUNS / warm_name(seed) / "ppo_warm_start.json").read_text(encoding="utf-8"))
    return r


def write_report(r: dict, path: Path) -> None:
    res, best, margin = r["results"], r["best"], r["margin"]

    def delta(name, ref):
        return "{:+.1f} ± {:.1f}".format(*paired_ci(res[name]["return"] - res[ref]["return"]))

    b = res[best]["return"]
    lines = [f"# Warm-start pilot (w_tardiness = {W:g})", "", PROTOCOL, "", "## Summary", "",
             f"Best rule, re-selected on training seeds: **{best}**, evaluation return {b.mean():.1f} ± {b.std():.1f}. "
             f"Teacher: edd_slack({margin:g}).", "",
             "| training seed | clone accuracy on its demonstrations | eval return before RL | eval return after RL "
             "| after − before RL, paired Δ | after RL − best rule, paired Δ | study PPO, eval return |",
             "|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        before, after, study = BEFORE.format(s), AFTER.format(s), STUDY.format(s)
        lines.append(f"| {s} | {100 * r['clone'][s]['accuracy']:.1f}% | {res[before]['return'].mean():.1f} "
                     f"| {res[after]['return'].mean():.1f} | {delta(after, before)} | {delta(after, best)} "
                     f"| {res[study]['return'].mean():.1f} |")
    k = r["warm_wins"]
    lines += ["", f"Decision rule: the warm start lets PPO beat the rules: **{'yes' if k >= 2 else 'no'}** ({k}/{len(SEEDS)} seeds).",
              "", "## Details", "", "Evaluation seeds, mean ± std:", "", summarize(res), "",
              "Paired against the best rule:", "", paired_summary(res, best), "",
              "After RL against the same seed's policy before RL and against the study's PPO:", "",
              "| pair | Δ return | Δ tardiness_min | Δ energy | higher return |", "|---|---|---|---|---|"]
    for s in SEEDS:
        after = AFTER.format(s)
        for ref in (BEFORE.format(s), STUDY.format(s)):
            row = paired_summary({ref: res[ref], after: res[after]}, ref).splitlines()[-1]
            lines.append(row.replace(f"| {after} |", f"| {after} vs {ref} |", 1))
    lines += ["", f"| policy | speed share low / nominal / high | launch order = EDD order | action = edd_slack({margin:g})'s "
              f"| most common launch order | training return, last {LAST_EPISODES} episodes |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        for name in (STUDY.format(s), BEFORE.format(s), AFTER.format(s)):
            bh = r["behaviour"][name]
            share = " / ".join(f"{100 * x:.1f}%" for x in bh["speed_share"])
            seq, count = bh["sequences"].most_common(1)[0]
            train = f"{r['train_return'][name]:.1f}" if name in r["train_return"] else "–"
            lines.append(f"| {name} | {share} | {bh['edd_order']}/{bh['episodes']} | {100 * bh['teacher_agreement']:.1f}% "
                         f"| {' → '.join(map(str, seq))} ({count}/{bh['episodes']}) | {train} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="skip training; use the runs already in results/runs/")
    ap.add_argument("--out", default="results/warm_start_pilot.md")
    args = ap.parse_args()
    missing = [run_name(W, s) for s in SEEDS if not (RUNS / run_name(W, s)).exists()]
    if missing:
        sys.exit(f"missing study runs {missing}: run scripts/run_sensitivity.py first")
    margin = tuned_margin()
    if not args.report_only:
        train_missing(args.steps, args.workers, margin)
    write_report(pilot(margin), Path(args.out))
    print(Path(args.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
