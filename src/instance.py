# This file constructs matching instances and computes match-based prices.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METERS_PER_MILE = 1609.0
MIN_SAVING_METERS = 1.0  # Minimum distance saving for pooling.


@dataclass(slots=True)
class Instance:
    solo_miles: np.ndarray  # d_i, shape (n,)
    arrival_rates: np.ndarray  # riders per minute, shape (n,)
    compatible: np.ndarray  # bool (n, n); diagonal True (same-type pairs)
    vehicle_miles: np.ndarray  # d_ij, shape (n, n); diagonal d_i
    allocated_miles: np.ndarray  # a_ij, shape (n, n); diagonal d_i / 2
    # Incompatible pairs have NaN payments.
    payments: np.ndarray  # (n, n + 1): solo column 0; partner j in j + 1.
    metadata: dict

    @property
    def n(self) -> int:
        return len(self.solo_miles)

    @property
    def dispatch_cost(self) -> float:
        return float(self.metadata["dispatch_cost_per_mile"])


def instance_name(detour_cap: float, cost: float) -> str:
    return f"d{round(100 * detour_cap):03d}_c{round(100 * cost):03d}"


def _route_candidates(L_diag, L_PP, L_PQ, L_QQ):
    # Distances in meters, shaped (4, n, n).
    # O = pickup; D = drop-off.
    d_i, d_j = L_diag[:, None], L_diag[None, :]
    total = np.stack(
        [
            L_PP + L_PQ.T + L_QQ,  # Oi Oj Di Dj
            L_PP + d_j + L_QQ.T,  # Oi Oj Dj Di
            L_PP.T + d_i + L_QQ,  # Oj Oi Di Dj
            L_PP.T + L_PQ + L_QQ.T,  # Oj Oi Dj Di
        ]
    )
    ride_i = np.stack([L_PP + L_PQ.T, total[1], np.broadcast_to(d_i, total[2].shape), L_PQ + L_QQ.T])
    ride_j = np.stack([L_PQ.T + L_QQ, np.broadcast_to(d_j, total[1].shape), total[2], L_PP.T + L_PQ])
    return total, ride_i, ride_j


def build_instance(
    blocks: dict, arrival_rates: np.ndarray, *, detour_cap: float, dispatch_cost: float, metadata: dict | None = None
) -> Instance:
    if not (detour_cap >= 0 and dispatch_cost >= 0):
        raise ValueError("detour_cap and dispatch_cost must be nonnegative")
    L_diag = np.asarray(blocks["L_diag"], float)
    n = len(L_diag)
    total, ride_i, ride_j = _route_candidates(L_diag, blocks["L_PP"], blocks["L_PQ"], blocks["L_QQ"])
    best = np.argmin(total, axis=0)  # Ties choose the first route.
    ii, jj = np.indices((n, n))
    route = total[best, ii, jj]
    detour = np.maximum(
        (ride_i[best, ii, jj] - L_diag[:, None]) / L_diag[:, None],
        (ride_j[best, ii, jj] - L_diag[None, :]) / L_diag[None, :],
    )
    saving = L_diag[:, None] + L_diag[None, :] - route
    within_cap = detour <= detour_cap + 1e-12
    ok = within_cap & (saving >= MIN_SAVING_METERS)
    compatible = ok | ok.T  # Symmetrize tied route choices.
    np.fill_diagonal(compatible, True)

    d = L_diag / METERS_PER_MILE
    vehicle = route / METERS_PER_MILE
    np.fill_diagonal(vehicle, d)
    alloc = 0.5 * (d[:, None] + vehicle - d[None, :])
    np.fill_diagonal(alloc, 0.5 * d)

    c = float(dispatch_cost)
    payments = np.full((n, n + 1), np.nan)  # Total payments in dollars.
    payments[:, 0] = (1.0 + c) / 2.0 * d
    payments[:, 1:] = np.where(compatible, (1.0 + c * alloc / d[:, None]) / 2.0 * d[:, None], np.nan)
    if np.any(payments[:, 1:] > payments[:, [0]] + 1e-12):
        raise AssertionError("a matched price exceeds the solo price")

    meta = dict(metadata or {})
    meta.update(
        {
            "n_types": n,
            "detour_cap": float(detour_cap),
            "dispatch_cost_per_mile": c,
            "meters_per_mile": METERS_PER_MILE,
            "min_saving_meters": MIN_SAVING_METERS,
            "compatible_pairs": int(np.triu(compatible, 1).sum()),
            # Feasible pairs with negligible savings.
            "relay_pairs_excluded": int(np.triu(within_cap & (np.abs(saving) < MIN_SAVING_METERS), 1).sum()),
            "total_arrival_rate_per_min": float(np.sum(arrival_rates)),
        }
    )
    return Instance(
        solo_miles=d,
        arrival_rates=np.asarray(arrival_rates, float),
        compatible=compatible,
        vehicle_miles=vehicle,
        allocated_miles=alloc,
        payments=payments,
        metadata=meta,
    )


def save_instance(inst: Instance, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "instance.npz",
        solo_miles=inst.solo_miles,
        arrival_rates=inst.arrival_rates,
        compatible=inst.compatible,
        vehicle_miles=inst.vehicle_miles,
        allocated_miles=inst.allocated_miles,
        payments=inst.payments,
    )
    (out / "metadata.json").write_text(json.dumps(inst.metadata, indent=2), newline="\n")


def load_instance(out_dir: str | Path) -> Instance:
    out = Path(out_dir)
    with np.load(out / "instance.npz") as a:
        arrays = {k: a[k] for k in a.files}
    return Instance(metadata=json.loads((out / "metadata.json").read_text()), **arrays)
