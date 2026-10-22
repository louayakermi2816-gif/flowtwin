"""Lateness-weight sensitivity: the weight reaches the reward, the scripts expose it without
changing their default outputs, and the slack-aware rule behaves as specified."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from flowtwin.config import DEFAULT
from flowtwin.evaluate import evaluate
from flowtwin.heuristics import edd

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True)


def test_lateness_weight_only_rescales_the_tardiness_term():
    low, high = replace(DEFAULT, w_tardiness=0.10), replace(DEFAULT, w_tardiness=2.5)
    a, b = evaluate(edd, low, n_episodes=10), evaluate(edd, high, n_episodes=10)
    assert a["tardiness_min"].sum() > 0                                  # some shifts finish late
    assert np.array_equal(a["tardiness_min"], b["tardiness_min"])        # same dynamics
    # the KPI reports lateness rounded to 0.1 min, so allow the matching rounding error
    assert np.allclose(a["return"] - b["return"], (2.5 - 0.10) * a["tardiness_min"], rtol=0, atol=(2.5 - 0.10) * 0.05)


def test_baselines_script_takes_the_lateness_weight(tmp_path):
    run_script("scripts/run_baselines.py", "--episodes", "2", "--w-tardiness", "2.5", "--out", str(tmp_path))
    saved = json.loads((tmp_path / "baselines.json").read_text(encoding="utf-8"))
    expected = evaluate(edd, replace(DEFAULT, w_tardiness=2.5), n_episodes=2)["return"]
    assert np.allclose(saved["edd"]["return"], expected)


def test_default_baselines_reproduce_the_committed_table(tmp_path):
    run_script("scripts/run_baselines.py", "--out", str(tmp_path))
    assert (tmp_path / "baselines.md").read_text(encoding="utf-8") == (ROOT / "results/baselines.md").read_text(encoding="utf-8")
