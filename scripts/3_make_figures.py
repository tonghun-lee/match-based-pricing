#!/usr/bin/env python3
# This file generates Figures 3 and 4 and supporting statistics.

from __future__ import annotations

import json
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instance import load_instance  # noqa: E402
from src.simulation import Simulator, make_mechanism, summarize  # noqa: E402

RUNS = ROOT / "results" / "runs" / "c070" / "w100"
POLICIES = ("myopic", "laissez_faire", "clearing")
LABELS = {"myopic": "Myopic", "laissez_faire": r"Strategic, $\sigma_\infty$", "clearing": r"Strategic, $\sigma_1$"}
SEED, REPLICATIONS, HORIZON, BURN_IN = 40042, 100, 1800.0, 180.0  # as in 1_run_experiments.py
WORKERS = 2
MIN_RIDERS = 1000  # hatch zones below this dispatch count
SERVICES = (("solo", "Solo dispatch"), ("same_type", "Same-type match"), ("cross_type", "Cross-type match"))
FILL = {"solo": "white", "same_type": "#bfbfbf", "cross_type": "white"}
HATCH = {"cross_type": "///"}
POLICY_STYLE = {  # color, line style
    "myopic": ("#0000ff", "-"),
    "laissez_faire": ("#ff0000", (0, (1, 1))),
    "clearing": ("#00ff00", (0, (2, 2))),
}

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral", "font.size": 18, "pdf.fonttype": 42})


class Recorder(Simulator):
    # Count offered and matched riders.
    def run(self, *args, **kwargs) -> dict:
        self.pairs = np.zeros((self.mech.n, self.mech.n))
        self.arrivals = self.offered = 0
        return super().run(*args, **kwargs)

    def on_offer(self, typ: int, offer: int) -> None:
        self.arrivals += 1
        self.offered += offer > 0

    def on_match(self, a, b, now: float) -> None:
        if now >= self.burn_in:
            self.pairs[a.typ, b.typ] += 1
            self.pairs[b.typ, a.typ] += 1


_RUN: dict = {}  # worker state


def _init_run(state: dict) -> None:
    _RUN.update(state)


def _replication(r: int) -> tuple[float, float, np.ndarray]:
    c = _RUN
    sim = Recorder(c["mech"], seed=SEED + 1009 * r, clearing=c["clearing"])
    out = sim.run(c["zeta"], c["gamma"], mode=c["mode"], horizon=HORIZON, burn_in=BURN_IN)
    return summarize(out["stats"], HORIZON - BURN_IN)["profit"], sim.offered / sim.arrivals, sim.pairs


def resimulate(policy: str, workers: int) -> tuple[float, np.ndarray]:
    # Replay evaluation seeds to recover offers and matches.
    run = json.loads((RUNS / policy / "summary.json").read_text())
    mech = make_mechanism(
        load_instance(ROOT / run["instance"]), theta=run["theta"], waiting_cost=run["w_per_hour"] / 60.0
    )
    with np.load(RUNS / policy / "learned.npz") as profile:
        zeta, gamma = profile["zeta"], profile["gamma"]
    mode = "myopic" if policy == "myopic" else "strategic"
    state = dict(mech=mech, zeta=zeta, gamma=gamma, mode=mode, clearing=policy == "clearing")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_run, initargs=(state,)) as ex:
        outs = list(ex.map(_replication, range(REPLICATIONS)))
    stored = pd.read_csv(RUNS / policy / "evaluation.csv", float_precision="round_trip")["profit"].to_numpy()
    if not np.array_equal([o[0] for o in outs], stored):
        raise RuntimeError(f"re-simulated profits differ from {RUNS / policy / 'evaluation.csv'}")
    return float(np.mean([o[1] for o in outs])), sum(o[2] for o in outs)


def profit_by_service(policy: str, pairs: np.ndarray, inst) -> dict:
    # Profit per minute by service type.
    c, P, d = inst.dispatch_cost, inst.payments, inst.solo_miles
    totals = pd.read_csv(RUNS / policy / "evaluation_types.csv")
    solo = (totals["completed"] - totals["matched"]).to_numpy()
    pair_profit = np.nan_to_num(P[:, 1:] + P[:, 1:].T - c * inst.vehicle_miles)
    minutes = REPLICATIONS * (HORIZON - BURN_IN)
    return {
        "solo": math.fsum(solo * (P[:, 0] - c * d)) / minutes,
        "same_type": math.fsum(np.diag(pairs) / 2 * np.diag(pair_profit)) / minutes,
        "cross_type": math.fsum((np.triu(pairs, 1) * pair_profit).ravel()) / minutes,
    }


def zone_map(ax, zones, values: pd.Series, cmap: str, vmin: float, vmax: float) -> None:
    df = pd.DataFrame({"cluster": range(len(zones)), "value": values.reindex(range(len(zones))).to_numpy()})
    zones.merge(df, on="cluster").plot(
        ax=ax,
        column="value",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        missing_kwds={"color": "lightgrey", "hatch": "////"},
    )
    ax.set_axis_off()


def save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_row(panels: list, pad: float = 0.04) -> None:
    # Align panel crops vertically.
    boxes = [fig.get_tightbbox(fig.canvas.get_renderer()) for fig, _ in panels]
    y0, y1 = min(b.y0 for b in boxes) - pad, max(b.y1 for b in boxes) + pad
    for (fig, path), b in zip(panels, boxes, strict=True):
        fig.savefig(path, bbox_inches=Bbox([[b.x0 - pad, y0], [b.x1 + pad, y1]]))
        plt.close(fig)


# Shared panel height and vertical plotting bounds.
FIG_HEIGHT, FRAME = 4.8, (0.17, 0.95)


def chart_axes(width: float = 6.4):
    fig = plt.figure(figsize=(width, FIG_HEIGHT))
    left, right = 0.95, 0.25  # inches
    ax = fig.add_axes((left / width, FRAME[0], 1 - (left + right) / width, FRAME[1] - FRAME[0]))
    return fig, ax


def plot_arrival_rate(zones, rate: pd.Series):
    x0, y0, x1, y1 = zones.total_bounds
    pad = 0.01 * (y1 - y0)
    height = (FRAME[1] - FRAME[0]) * FIG_HEIGHT
    map_w = height * (x1 - x0 + 2 * pad) / (y1 - y0 + 2 * pad)
    left, gap, bar, right = 0.05, 0.25, 0.2, 1.0  # inches
    width = left + map_w + gap + bar + right
    fig = plt.figure(figsize=(width, FIG_HEIGHT))
    ax = fig.add_axes((left / width, FRAME[0], map_w / width, FRAME[1] - FRAME[0]))
    cax = fig.add_axes(((left + map_w + gap) / width, FRAME[0], bar / width, FRAME[1] - FRAME[0]))
    zone_map(ax, zones, rate, "Greens", 0.0, rate.max())
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0.0, rate.max()), "Greens"), cax=cax).set_label("Riders (per min)")
    return fig


def plot_policy_maps(zones, values: dict, cmap: str, label: str, path: Path, ticks=None) -> None:
    # Use a shared color scale.
    vmin = min(np.nanmin(v) for v in values.values())
    vmax = max(np.nanmax(v) for v in values.values())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), gridspec_kw={"wspace": 0.02})
    for ax, policy in zip(axes, ("laissez_faire", "clearing"), strict=True):
        zone_map(ax, zones, values[policy], cmap, vmin, vmax)
        ax.text(0.5, -0.02, LABELS[policy], transform=ax.transAxes, ha="center", va="top")
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    fig.colorbar(sm, ax=list(axes), shrink=0.88, fraction=0.05, pad=0.03, ticks=ticks).set_label(label)
    save(fig, path)


def plot_profit(profit: pd.DataFrame):
    fig, ax = chart_axes()
    xs, width = np.arange(len(profit)), 0.26
    for k, (col, label) in enumerate(SERVICES):
        ax.bar(
            xs + (k - 1) * (width + 0.02),
            profit[col],
            width=width,
            color=FILL[col],
            edgecolor="black",
            hatch=HATCH.get(col),
            linewidth=1.0,
            label=label,
            zorder=3,
        )
    ax.set_xticks(xs, [LABELS[p] for p in profit.index])
    ax.set_ylabel(r"Profit (\$/min)")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=16)
    ax.set_ylim(0, 3.0)
    ax.set_yticks(np.arange(0, 3.01, 0.5))
    ax.spines["left"].set_bounds(0, 3)
    return fig


def plot_same_type(inst, pairs: dict):
    # Accumulate matches from highest to lowest arrival rate.
    order = np.argsort(-inst.arrival_rates, kind="stable")
    x = np.concatenate([[0.0], np.cumsum(inst.arrival_rates[order])])
    minutes = REPLICATIONS * (HORIZON - BURN_IN)
    fig, ax = chart_axes(width=5.4)
    for policy in POLICIES:
        color, style = POLICY_STYLE[policy]
        y = np.concatenate([[0.0], np.cumsum(np.diag(pairs[policy])[order]) / minutes])
        ax.plot(x, y, color=color, linestyle=style, linewidth=2.5, label=LABELS[policy], clip_on=False, zorder=3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 3.5)  # headroom for the legend
    ax.set_yticks(np.arange(0, 3.51, 0.5))
    ax.set_xlabel("Cumulative arrivals (per min)")
    ax.set_ylabel("Cumulative matched riders (per min)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=16)
    return fig


def main() -> None:
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    inst = load_instance(ROOT / json.loads((RUNS / "clearing" / "summary.json").read_text())["instance"])
    types = pd.read_csv(ROOT / "results" / "instances" / "rider_types.csv")
    with open(ROOT / "data" / "Chicago_zones.pickle", "rb") as f:
        zones = pickle.load(f)["clusters_gdf"]

    offered, pairs = {}, {}
    for policy in POLICIES:
        offered[policy], pairs[policy] = resimulate(policy, WORKERS)
    profit = pd.DataFrame({policy: profit_by_service(policy, pairs[policy], inst) for policy in POLICIES}).T
    table1 = {p: pd.read_csv(RUNS / p / "evaluation.csv")["profit"].mean() for p in POLICIES}
    if not np.allclose(profit.sum(axis=1).to_numpy(), [table1[p] for p in POLICIES], rtol=1e-9):
        raise RuntimeError("profit by service does not add up to the Table 1 profit")

    rate = types.groupby("pickup_zone")["arrival_rate_per_min"].sum()
    totals = {p: pd.read_csv(RUNS / p / "evaluation_types.csv").groupby(types["pickup_zone"]).sum() for p in POLICIES}
    sparse = np.minimum(totals["laissez_faire"]["completed"], totals["clearing"]["completed"]) < MIN_RIDERS
    delayed = {p: (totals[p]["waited"] / totals[p]["completed"]).where(~sparse) for p in POLICIES[1:]}
    discount = {p: (100 * totals[p]["discount"] / totals[p]["joined"]).where(~sparse) for p in POLICIES[1:]}

    save_row(
        [
            (plot_arrival_rate(zones, rate), out / "zones_arrival_rate.pdf"),
            (plot_same_type(inst, pairs), out / "same_type_matches.pdf"),
            (plot_profit(profit), out / "profit_by_service.pdf"),
        ]
    )
    plot_policy_maps(zones, delayed, "Blues", "Fraction of requests", out / "panel_delay.pdf")
    plot_policy_maps(
        zones, discount, "Reds", "Discount (% of solo price)", out / "panel_threshold.pdf", ticks=[0, 3, 6, 9, 12]
    )

    order = np.argsort(-inst.arrival_rates, kind="stable")[:10]
    same_type = {p: np.diag(pairs[p]) / (REPLICATIONS * (HORIZON - BURN_IN)) for p in POLICIES}  # riders/min
    stats = {
        "cost": 0.7,
        "w": 1.0,
        "share_of_arrivals_offered_a_match": offered,
        "ten_densest_types": {
            "share_of_arrivals": float(inst.arrival_rates[order].sum() / inst.arrival_rates.sum()),
            "share_of_same_type_matches_under_sigma_1": float(
                np.diag(pairs["clearing"])[order].sum() / np.diag(pairs["clearing"]).sum()
            ),
            "same_type_matched_riders_per_minute": {p: float(v[order].sum()) for p, v in same_type.items()},
        },
        "same_type_matched_riders_per_minute": {p: float(v.sum()) for p, v in same_type.items()},
        "profit_by_service_per_minute": profit.to_dict(orient="index"),
        "hatched_zones": [int(z) for z in sparse.index[sparse]],
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), newline="\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
