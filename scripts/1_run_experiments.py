#!/usr/bin/env python3
# This file runs the Chicago experiments and evaluates rider strategies.

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chicago_data import build_rider_types, load_trips  # noqa: E402
from src.instance import build_instance, instance_name, load_instance, save_instance  # noqa: E402
from src.learning import LearningConfig, train  # noqa: E402
from src.network import distance_blocks, load_network  # noqa: E402
from src.simulation import evaluate, make_mechanism  # noqa: E402

DETOUR_CAP = 0.25
COSTS = (0.7, 0.9)  # $/vehicle-mile
WAITING_COSTS = (0.5, 1.0, 1.5, 2.0)  # $/solo-mile/hour
POLICIES = ("myopic", "laissez_faire", "clearing")
THETA = 1.0  # window expiry rate per minute
SEED = 42
EVAL_SEED, REPLICATIONS = 40042, 100
RESULTS = ROOT / "results"


def run_dir(c: float, w: float, policy: str) -> Path:
    return RESULTS / "runs" / f"c{round(100 * c):03d}" / f"w{round(100 * w):03d}" / policy


def build_instances() -> None:
    data, out = ROOT / "data", RESULTS / "instances"
    trips = load_trips(data / "Chicago_data.csv")
    G = load_network(data / "Chicago_network.pickle")
    types = build_rider_types(trips, G, data / "Chicago_zones.pickle", n_types=244, seed=0)
    out.mkdir(parents=True, exist_ok=True)
    types.centers.to_csv(out / "rider_types.csv", index=False, lineterminator="\n")
    blocks = distance_blocks(G, types.centers["cluster_pickup_node"], types.centers["cluster_dropoff_node"])
    meta = {"trips": len(trips), "service_dates": types.service_dates, "kmeans_seed": 0}
    for c in COSTS:
        inst = build_instance(blocks, types.arrival_rates, detour_cap=DETOUR_CAP, dispatch_cost=c, metadata=meta)
        save_instance(inst, out / instance_name(DETOUR_CAP, c))


def learn(c: float, w: float, policy: str, workers: int) -> bool:
    out = run_dir(c, w, policy)
    out.mkdir(parents=True, exist_ok=True)
    instance_dir = RESULTS / "instances" / instance_name(DETOUR_CAP, c)
    inst = load_instance(instance_dir)
    mech = make_mechanism(inst, theta=THETA, waiting_cost=w / 60.0)
    if policy == "myopic":
        cfg = LearningConfig(mode="myopic", seed=SEED + 17, workers=workers)
    else:
        cfg = LearningConfig(mode="strategic", clearing=policy == "clearing", seed=SEED, workers=workers)
    t0 = time.time()
    label = "/".join(out.parts[-3:])
    res = train(mech, cfg, log=lambda s: print(f"{label} {s}", flush=True))
    extra = {} if policy == "myopic" else {"weights": res["weights"], "played_n": res["played_n"]}
    np.savez_compressed(out / "learned.npz", zeta=res["zeta"], gamma=res["gamma"], **extra)
    pd.DataFrame(res["history"]).to_csv(out / "history.csv", index=False, lineterminator="\n")
    summary = {
        "instance": instance_dir.relative_to(ROOT).as_posix(),
        "instance_metadata": inst.metadata,
        "policy": policy,
        "w_per_hour": w,
        "theta": THETA,
        "learning": {k: v for k, v in asdict(cfg).items() if k != "workers"},
        "certified": res["certified"],
        "outer_iterations": len(res["history"]),
        "certificate": res["certificate"],
        "seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), newline="\n")
    return res["certified"]


def evaluate_profile(c: float, w: float, policy: str, workers: int) -> None:
    out = run_dir(c, w, policy)
    run = json.loads((out / "summary.json").read_text())
    mech = make_mechanism(load_instance(ROOT / run["instance"]), theta=run["theta"], waiting_cost=w / 60.0)
    with np.load(out / "learned.npz") as profile:
        zeta, gamma = profile["zeta"], profile["gamma"]
    mode = "myopic" if policy == "myopic" else "strategic"
    rows, types = evaluate(
        mech,
        zeta,
        gamma,
        mode=mode,
        clearing=policy == "clearing",
        seed=EVAL_SEED,
        replications=REPLICATIONS,
        workers=workers,
    )
    pd.DataFrame(rows).to_csv(out / "evaluation.csv", index=False, lineterminator="\n")
    pd.DataFrame({"type": np.arange(mech.n), **types}).to_csv(
        out / "evaluation_types.csv", index=False, lineterminator="\n"
    )


def run_profile(task: tuple) -> str:
    c, w, policy, workers = task
    out = run_dir(c, w, policy)
    summary = out / "summary.json"
    if not (summary.exists() and json.loads(summary.read_text())["certified"]):
        if not learn(c, w, policy, workers):
            return f"{out}: not certified"
    if not (out / "evaluation.csv").exists():
        evaluate_profile(c, w, policy, workers)
    return f"{out}: done"


def collect() -> pd.DataFrame:
    # Aggregate means and standard errors.
    rows = []
    for c in COSTS:
        for w in WAITING_COSTS:
            for policy in POLICIES:
                out = run_dir(c, w, policy)
                if not (out / "evaluation.csv").exists():
                    continue
                run = json.loads((out / "summary.json").read_text())
                ev = pd.read_csv(out / "evaluation.csv", float_precision="round_trip").drop(columns="replication")
                row = {
                    "cost": c,
                    "w": w,
                    "policy": policy,
                    "certified": run["certified"],
                    "outer_iterations": run["outer_iterations"],
                    "replications": len(ev),
                }
                row.update(ev.mean().to_dict())
                # Use each metric's nonmissing replications.
                row.update({f"{k}_se": v for k, v in (ev.std(ddof=1) / np.sqrt(ev.notna().sum())).items()})
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the Chicago instances, then learn and evaluate the 24 strategy profiles behind Table 1."
    )
    p.add_argument("--jobs", type=int, default=2, help="profiles learned in parallel (one core each)")
    a = p.parse_args()

    if not all((RESULTS / "instances" / instance_name(DETOUR_CAP, c) / "instance.npz").exists() for c in COSTS):
        build_instances()
    workers = 1 if a.jobs > 1 else 2
    tasks = [(c, w, policy, workers) for c in COSTS for w in WAITING_COSTS for policy in POLICIES]
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            messages = list(ex.map(run_profile, tasks, chunksize=1))
    else:
        messages = [run_profile(t) for t in tasks]
    print("\n".join(messages))
    df = collect()
    df.to_csv(RESULTS / "table1_metrics.csv", index=False, lineterminator="\n")
    print(f"wrote {RESULTS / 'table1_metrics.csv'} ({len(df)} profiles)")
    sys.exit(0 if len(df) == len(tasks) else 1)


if __name__ == "__main__":
    main()
