"""Seeded evaluation of scheduling policies and Markdown results tables.

Every policy is evaluated on the same seeds (same orders, same breakdown times), so
policies can also be compared seed by seed: paired_summary() reports the mean
per-seed difference with a 95% confidence interval, which removes the demand
variance that dominates the per-policy standard deviation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .config import DEFAULT, LineConfig
from .env import FlowTwinEnv

KPI_COLUMNS = ["return", "units_out", "throughput_per_h", "changeover_min", "energy", "tardiness_min", "unmet_units",
               "invalid_actions", "time_min"]


def run_episode(env: FlowTwinEnv, policy: Callable, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)        # policies that sample the action space (random) are reproducible too
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        total += r
        done = term or trunc
    k = info["kpis"]
    return {"return": total, **{c: k[c] for c in KPI_COLUMNS if c in k}}


def evaluate(policy: Callable, cfg: LineConfig = DEFAULT, n_episodes: int = 30, seed0: int = 1000) -> Dict[str, np.ndarray]:
    env = FlowTwinEnv(cfg)
    rows = [run_episode(env, policy, seed0 + i) for i in range(n_episodes)]
    return {c: np.array([r[c] for r in rows], dtype=float) for c in KPI_COLUMNS}


def summarize(results: Dict[str, Dict[str, np.ndarray]]) -> str:
    """results: policy name -> evaluate() output. Returns a Markdown table (mean +/- std)."""
    cols = ["return", "units_out", "throughput_per_h", "changeover_min", "energy", "tardiness_min", "unmet_units",
            "invalid_actions"]
    head = "| policy | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for name, res in results.items():
        cells = [f"{res[c].mean():.1f} ± {res[c].std():.1f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def t95(df: int) -> float:
    """Two-sided 95% Student-t quantile, Cornish-Fisher expansion (Abramowitz & Stegun 26.7.5);
    within 0.01 of the exact value for df >= 4 and within 1e-5 for df >= 20 (2.0452 for 30 episodes)."""
    z = 1.959964
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


TIE_TOL = 1e-6   # two policies that schedule a shift identically can differ by float noise (~1e-13)


def paired_summary(results: Dict[str, Dict[str, np.ndarray]], ref: str,
                   cols: Sequence[str] = ("return", "tardiness_min", "energy")) -> str:
    """Markdown table of per-seed differences (policy - ref): mean ± 95% CI half-width, and the
    number of seeds on which the policy's return is higher than the reference's (ties, i.e.
    differences within TIE_TOL, are counted separately)."""
    base = results[ref]
    n = len(base["return"])
    head = f"| paired vs {ref} | " + " | ".join(f"Δ {c}" for c in cols) + " | higher return |"
    lines = [head, "|" + "---|" * (len(cols) + 2)]
    for name, res in results.items():
        if name == ref:
            continue
        cells = []
        for c in cols:
            d = res[c] - base[c]
            half = t95(n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            cells.append(f"{d.mean():+.1f} ± {half:.1f}")
        d_ret = res["return"] - base["return"]
        higher, ties = int((d_ret > TIE_TOL).sum()), int((np.abs(d_ret) <= TIE_TOL).sum())
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {higher}/{n}" + (f" ({ties} tied)" if ties else "") + " |")
    return "\n".join(lines)


def save_results(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, name: str, refs: Sequence[str] = ()) -> None:
    """Write <name>.md (mean ± std table, then one paired table per reference policy), .csv and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = [summarize(results)] + [paired_summary(results, r) for r in refs if r in results]
    (out_dir / f"{name}.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "episode"] + KPI_COLUMNS)
        for pname, res in results.items():
            for i in range(len(res["return"])):
                w.writerow([pname, i] + [float(res[c][i]) for c in KPI_COLUMNS])
    (out_dir / f"{name}.json").write_text(json.dumps({p: {c: v.tolist() for c, v in r.items()} for p, r in results.items()}, indent=1), encoding="utf-8")
