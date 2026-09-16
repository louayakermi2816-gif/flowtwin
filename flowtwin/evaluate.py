"""Seeded evaluation of scheduling policies and a Markdown results table."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .config import DEFAULT, LineConfig
from .env import FlowTwinEnv

KPI_COLUMNS = ["return", "units_out", "throughput_per_h", "changeover_min", "energy", "tardiness_min", "unmet_units", "time_min"]


def run_episode(env: FlowTwinEnv, policy: Callable, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
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
    cols = ["return", "units_out", "throughput_per_h", "changeover_min", "energy", "tardiness_min", "unmet_units"]
    head = "| policy | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for name, res in results.items():
        cells = [f"{res[c].mean():.1f} ± {res[c].std():.1f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(summarize(results) + "\n", encoding="utf-8")
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "episode"] + KPI_COLUMNS)
        for pname, res in results.items():
            for i in range(len(res["return"])):
                w.writerow([pname, i] + [float(res[c][i]) for c in KPI_COLUMNS])
    (out_dir / f"{name}.json").write_text(json.dumps({p: {c: v.tolist() for c, v in r.items()} for p, r in results.items()}, indent=1), encoding="utf-8")
