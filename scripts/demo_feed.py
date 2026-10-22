"""Drive the FlowTwin API so the dashboard has something to show: start a shift, step it with a
policy at a steady pace, and start the next shift when it ends. Standard library only.

    python scripts/demo_feed.py --policy ppo --interval 0.5
    python scripts/demo_feed.py --policy edd --shifts 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def call(api: str, method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--policy", default="ppo", help="ppo, dqn or a rule: edd, edd_low, fifo, spt")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between decisions")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--shifts", type=int, default=0, help="number of shifts; 0 runs until interrupted")
    args = ap.parse_args()

    policy = args.policy
    models = call(args.api, "GET", "/health").get("models", [])
    if policy in ("ppo", "dqn") and policy not in models:
        print(f"the API has no {policy} model loaded (loaded: {models or 'none'}); using edd instead")
        policy = "edd"

    shift = 0
    while args.shifts == 0 or shift < args.shifts:
        seed = args.seed0 + shift
        call(args.api, "POST", f"/twin/reset?seed={seed}")
        print(f"shift {shift + 1} (seed {seed}) with {policy}")
        while True:
            k = call(args.api, "POST", "/twin/step", {"policy": policy})["kpis"]
            print(f"  t={k['time_min']:6.1f} min  units={k['units_out']:4d}  late={k['tardiness_min']:6.1f} min", flush=True)
            if k["finished"]:
                break
            time.sleep(args.interval)
        shift += 1


if __name__ == "__main__":
    main()
