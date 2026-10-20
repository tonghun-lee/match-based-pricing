# This file learns and validates rider strategies.

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .simulation import Mechanism, Simulator

PRICE_TOL = 1e-12


class ActionIndex:
    # Threshold groups are stored consecutively for each (type, offer).

    def __init__(self, mech: Mechanism) -> None:
        n, P = mech.n, mech.payments
        self.cell_offset = np.full((n, n + 1), -1, dtype=np.int64)
        self.cell_size = np.zeros((n, n + 1), dtype=np.int64)
        self.reps: dict[tuple[int, int], np.ndarray] = {}  # Representative labels.
        self.members: dict[tuple[int, int], list[np.ndarray]] = {}
        self.label_group: dict[tuple[int, int], np.ndarray] = {}  # Label-to-group map; -1 means inactive.
        self.cells: list[tuple[int, int]] = []
        offset = 0
        for i in range(n):
            labels = mech.labels[i]
            prices = P[i, labels]
            for ell in map(int, labels):
                p_offer = float(P[i, ell])
                # Group 0 requests now; waiting groups follow increasing prices.
                groups, reps = [labels[prices >= p_offer - PRICE_TOL]], [ell]
                wait = labels[prices < p_offer - PRICE_TOL]
                wait = wait[np.argsort(P[i, wait], kind="stable")]
                wp = P[i, wait]
                start = 0
                for k in range(1, len(wait) + 1):
                    if k == len(wait) or wp[k] > wp[start] + PRICE_TOL:
                        groups.append(np.sort(wait[start:k]))
                        reps.append(int(groups[-1][0]))
                        start = k
                lab2g = np.full(n + 1, -1, dtype=np.int64)
                for g, grp in enumerate(groups):
                    lab2g[grp] = g
                key = (i, ell)
                self.reps[key], self.members[key], self.label_group[key] = np.asarray(reps), groups, lab2g
                self.cell_offset[i, ell], self.cell_size[i, ell] = offset, len(groups)
                self.cells.append(key)
                offset += len(groups)
        self.size = int(offset)
        self.rate = P / np.maximum(mech.solo_miles, 1e-9)[:, None]  # Dollars per solo mile.

    def now_value(self, i: int, ell: int) -> float | None:
        # Match offers dispatch immediately.
        return None if ell == 0 else float(self.rate[i, ell])

    def flat(self, i: int, q: int, ell: int) -> int:
        g = int(self.label_group[(i, ell)][q])
        if g < 0:
            raise KeyError(f"threshold {q} is not active for cell ({i}, {ell})")
        return int(self.cell_offset[i, ell]) + g

    def slice(self, i: int, ell: int) -> slice:
        o = int(self.cell_offset[i, ell])
        return slice(o, o + int(self.cell_size[i, ell]))


def initial_zeta(mech: Mechanism, value: float) -> np.ndarray:
    zeta = np.zeros((mech.n, mech.n + 1))
    for i in range(mech.n):
        zeta[i, mech.labels[i]] = value
    return zeta


def myopic_gamma(mech: Mechanism) -> np.ndarray:
    # Threshold 0 always requests immediately.
    gamma = np.zeros((mech.n, mech.n + 1, mech.n + 1))
    for i in range(mech.n):
        gamma[i, 0, mech.labels[i]] = 1.0
    return gamma


def initial_gamma(mech: Mechanism, index: ActionIndex, exploration: float) -> np.ndarray:
    # Combine immediate requests with uniformly random thresholds.
    gamma = myopic_gamma(mech)
    for i in range(mech.n):
        labels = mech.labels[i]
        for ell in labels:
            gamma[i, labels, ell] = (1.0 - exploration) * gamma[i, labels, ell] + exploration / len(labels)
    out = np.zeros_like(gamma)
    for i, ell in index.cells:
        mass = np.asarray([gamma[i, grp, ell].sum() for grp in index.members[(i, ell)]])
        out[i, index.reps[(i, ell)], ell] = mass / mass.sum()
    return out


def played_strategy(index: ActionIndex, weights: np.ndarray, act: RunningStats, n_use: float) -> np.ndarray:
    # Exclude waiting groups with fewer than n_use samples.
    n_tot = act.n[0] + act.n[1]
    gamma = weights.copy()
    for i, ell in index.cells:
        sl = index.slice(i, ell)
        if sl.stop - sl.start == 1:
            continue
        reps = index.reps[(i, ell)]
        p = weights[i, reps, ell]
        p[1:][n_tot[sl][1:] < n_use] = 0.0
        s = p.sum()
        if s > 0:
            p /= s
        else:
            p[:] = 0.0
            p[0] = 1.0
        gamma[i, reps, ell] = p
    return gamma


class LearningSimulator(Simulator):
    # Split disutility samples by rider ID parity.

    def __init__(self, mech: Mechanism, index: ActionIndex, *, probe_share: float, seed: int, clearing: bool) -> None:
        super().__init__(mech, seed=seed, clearing=clearing)
        self.index = index
        self.probe_share = float(probe_share)
        self.window = (0.0, float("inf"))
        self.reset()

    def reset(self) -> None:
        n, S = self.mech.n, self.index.size
        # Threshold statistics: (half, group).
        self.act_sum, self.act_sq, self.act_cnt = (np.zeros((2, S)) for _ in range(3))
        # Non-probe statistics: (type, offer).
        self.pop_sum, self.pop_sq, self.pop_cnt = (np.zeros((n, n + 1)) for _ in range(3))
        self.offers = np.zeros((n, n + 1))

    def new_rider(self, typ, threshold, offer, now):
        probe = self.probe_share > 0.0 and self.rng.random() < self.probe_share
        if probe:
            reps = self.index.reps[(typ, offer)]
            threshold = int(reps[int(self.rng.integers(len(reps)))])
        ag = super().new_rider(typ, threshold, offer, now)
        ag.probe = probe
        return ag

    def on_offer(self, typ, offer):
        self.offers[typ, offer] += 1.0

    def on_departure(self, ag, payment, now):
        if self.window[0] <= ag.join_time <= self.window[1]:
            self.record(ag, payment, now)

    def record(self, ag, payment: float, now: float) -> None:
        typ, ell, d = ag.typ, ag.offer, self.miles[ag.typ]
        u = (payment + self.mech.waiting_cost * d * (now - ag.join_time)) / d  # Dollars per solo mile.
        f, h = self.index.flat(typ, ag.threshold, ell), ag.rider_id & 1
        self.act_sum[h, f] += u
        self.act_sq[h, f] += u * u
        self.act_cnt[h, f] += 1.0
        if not ag.probe:
            self.pop_sum[typ, ell] += u
            self.pop_sq[typ, ell] += u * u
            self.pop_cnt[typ, ell] += 1.0


def run_truncated(
    sim: LearningSimulator, zeta, gamma, *, mode: str, horizon: float, burn_in: float, tail: float
) -> dict:
    sim.reset()
    sim.window = (float(burn_in), float(horizon) - float(tail))
    out = sim.run(zeta, gamma, mode=mode, horizon=horizon, burn_in=burn_in)
    unfinished = 0
    for ag in out["shadow"] + [a for a in out["actual"] if a is not None]:
        if sim.window[0] <= ag.join_time <= sim.window[1]:
            unfinished += 1
            # Unfinished riders use minimum payment plus accrued waiting costs.
            sim.record(ag, float(np.min(sim.mech.payments[ag.typ, sim.mech.labels[ag.typ]])), horizon)
    keys = ("act_sum", "act_sq", "act_cnt", "pop_sum", "pop_sq", "pop_cnt", "offers")
    return {
        **{k: getattr(sim, k).copy() for k in keys},
        "unfinished": float(unfinished),
        "profit": out["stats"]["profit"] / (horizon - burn_in),
    }


_AUDIT: dict = {}  # Audit state for each worker.


def _init_audit(state: dict) -> None:
    _AUDIT.clear()
    _AUDIT.update(state)


def _audit_one(seed: int) -> dict:
    c = _AUDIT
    sim = LearningSimulator(c["mech"], c["index"], probe_share=c["probe_share"], seed=int(seed), clearing=c["clearing"])
    return run_truncated(
        sim, c["zeta"], c["gamma"], mode=c["mode"], horizon=c["horizon"], burn_in=c["burn_in"], tail=c["tail"]
    )


def run_audit(mech, index, zeta, gamma, *, mode, clearing, probe_share, seeds, horizon, burn_in, tail, workers) -> dict:
    # Simulate independent replications with the profile fixed.
    state = dict(
        mech=mech,
        index=index,
        zeta=zeta,
        gamma=gamma,
        mode=mode,
        clearing=clearing,
        probe_share=probe_share,
        horizon=horizon,
        burn_in=burn_in,
        tail=tail,
    )
    seeds = [int(s) for s in seeds]
    if workers <= 1:
        _init_audit(state)
        return _total(map(_audit_one, seeds))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_audit, initargs=(state,)) as ex:
        return _total(ex.map(_audit_one, seeds, chunksize=1))


def _total(outs) -> dict:
    acc = None
    for out in outs:
        acc = dict(out) if acc is None else {k: acc[k] + v for k, v in out.items()}
    return acc


@dataclass
class RunningStats:
    # Decayed estimates with capped sample counts.

    shape: tuple
    cap: float
    decay: float
    n: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.n, self.mean, self.m2 = np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape)

    def update(self, s: np.ndarray, sq: np.ndarray, c: np.ndarray) -> None:
        self.n *= self.decay
        tot = self.n + c
        upd = c > 0
        self.mean[upd] = (self.n[upd] * self.mean[upd] + s[upd]) / tot[upd]
        self.m2[upd] = (self.n[upd] * self.m2[upd] + sq[upd]) / tot[upd]
        self.n = np.minimum(tot, self.cap)

    def state(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load(self, st: dict) -> None:
        self.n, self.mean, self.m2 = (np.asarray(st[k], dtype=float).copy() for k in ("n", "mean", "m2"))


class Evidence(NamedTuple):
    # Threshold statistics by half; non-probe statistics by (type, offer).

    act_n: np.ndarray
    act_mean: np.ndarray
    act_var: np.ndarray | None  # Optional for standard errors.
    pop_n: np.ndarray
    pop_mean: np.ndarray
    pop_var: np.ndarray | None


def running_evidence(act: RunningStats, pop: RunningStats) -> Evidence:
    return Evidence(act.n, act.mean, None, pop.n, pop.mean, None)


def audit_evidence(agg: dict) -> Evidence:
    cnt, pc = agg["act_cnt"], agg["pop_cnt"]
    act_mean = agg["act_sum"] / np.maximum(cnt, 1)
    pop_mean = agg["pop_sum"] / np.maximum(pc, 1)
    return Evidence(
        cnt,
        act_mean,
        np.maximum(agg["act_sq"] / np.maximum(cnt, 1) - act_mean**2, 0.0),
        pc,
        pop_mean,
        np.maximum(agg["pop_sq"] / np.maximum(pc, 1) - pop_mean**2, 0.0),
    )


def br_value(nA, mA, nB, mB, n_half: int):
    # Choose each half's best group; evaluate on the other.
    ok = (nA >= n_half) & (nB >= n_half)
    if not np.any(ok):
        return None
    idx = np.flatnonzero(ok)
    gA, gB = idx[int(np.argmin(mA[idx]))], idx[int(np.argmin(mB[idx]))]
    return 0.5 * (float(mB[gA]) + float(mA[gB])), (int(gA), int(gB))


def cell_value(
    index: ActionIndex, i: int, ell: int, ev: Evidence, *, mode: str, n_half: int, n_pop: int, prior_n: float = 10.0
):
    # Return per-mile joining value, standard error, and estimate source.
    cands = []
    v_now = index.now_value(i, ell)
    if v_now is not None:
        cands.append((v_now, 0.0, "exact"))
    if mode == "strategic":
        sl = index.slice(i, ell)
        res = br_value(ev.act_n[0][sl], ev.act_mean[0][sl], ev.act_n[1][sl], ev.act_mean[1][sl], n_half)
        if res is not None:
            ga, gb = res[1]
            se = 0.0
            if ev.act_var is not None:
                se = 0.5 * math.sqrt(
                    ev.act_var[1][sl][ga] / max(ev.act_n[1][sl][ga], 1)
                    + ev.act_var[0][sl][gb] / max(ev.act_n[0][sl][gb], 1)
                )
            cands.append((res[0], se, "br"))
    if cands:
        return min(cands, key=lambda c: c[0])
    n = float(ev.pop_n[i, ell])
    if n >= n_pop:
        se = math.sqrt(float(ev.pop_var[i, ell]) / n) if ev.pop_var is not None else 0.0
        return float(ev.pop_mean[i, ell]), se, "pop"
    # Shrink sparse estimates toward the solo price.
    prior = float(index.rate[i, 0])
    return (n * float(ev.pop_mean[i, ell]) + prior_n * prior) / (n + prior_n), float("nan"), "prior"


def entry_targets(index: ActionIndex, ev: Evidence, *, mode: str, n_half: int, n_pop: int) -> np.ndarray:
    target = np.full(index.cell_offset.shape, np.nan)
    for i, ell in index.cells:
        V, _, _ = cell_value(index, i, ell, ev, mode=mode, n_half=n_half, n_pop=n_pop)
        target[i, ell] = min(max(1.0 - V, 0.0), 1.0)
    return target


def mw_update(
    index: ActionIndex,
    weights: np.ndarray,
    act: RunningStats,
    pop: RunningStats,
    *,
    beta: float,
    n_use: float,
    floor: float,
) -> float:
    # Update group weights and return the largest change.
    n_tot = act.n[0] + act.n[1]
    m_pool = (act.n[0] * act.mean[0] + act.n[1] * act.mean[1]) / np.maximum(n_tot, 1e-12)
    delta = 0.0
    for i, ell in index.cells:
        sl = index.slice(i, ell)
        g = sl.stop - sl.start
        if g == 1:
            continue
        cnt, m = n_tot[sl].copy(), m_pool[sl].copy()
        v_now = index.now_value(i, ell)
        if v_now is not None:
            m[0], cnt[0] = v_now, np.inf  # Exact request-now value.
        sampled = cnt >= n_use
        if not np.any(sampled):
            continue
        # Sparse groups use the population mean or best sampled value.
        fill = float(pop.mean[i, ell]) if pop.n[i, ell] >= n_use else float(np.min(m[sampled]))
        c = np.where(sampled, m, fill)
        reps = index.reps[(i, ell)]
        old = weights[i, reps, ell]
        w = old * np.exp(-beta * (c - c.min()))
        s = w.sum()
        p = np.full(g, 1.0 / g) if (not np.isfinite(s) or s <= 0) else w / s
        p = (1.0 - floor) * p + floor / g
        weights[i, reps, ell] = p
        delta = max(delta, float(np.max(np.abs(p - old))))
    return delta


def certificate_stats(
    index: ActionIndex, zeta: np.ndarray, agg: dict, *, mode: str, n_half: int, n_pop: int, n_cap: int, eps_cell: float
) -> dict:
    # Entry errors use offer weights; regret uses non-probe joiners.
    ev = audit_evidence(agg)
    offers = agg["offers"]
    w_cov = res_sum = abs_sum = exact_viol = exact_n = 0.0
    for i, ell in index.cells:
        V_now = index.now_value(i, ell)
        if V_now is not None:
            gap = float(zeta[i, ell]) - min(max(1.0 - V_now, 0.0), 1.0)
            exact_n += 1.0
            exact_viol += float(abs(gap) > 0.01 if mode == "myopic" else gap < -0.01)
        w = float(offers[i, ell])
        if w <= 0:
            continue
        V, _, src = cell_value(index, i, ell, ev, mode=mode, n_half=n_half, n_pop=n_pop)
        if src == "prior":
            continue
        r = float(zeta[i, ell]) - min(max(1.0 - V, 0.0), 1.0)
        w_cov += w
        res_sum += w * r
        abs_sum += w * abs(r)
    stats = {
        "entry_offer_coverage": w_cov / max(float(offers.sum()), 1e-12),
        "entry_signed_residual": res_sum / max(w_cov, 1e-12),
        "entry_mean_abs_residual": abs_sum / max(w_cov, 1e-12),
        "exact_bound_violation_share": exact_viol / max(exact_n, 1.0),
        "unfinished_share": float(agg["unfinished"]) / max(float(np.sum(ev.act_n)), 1.0),
    }
    if mode == "strategic":
        pc = ev.pop_n
        mass = max(float(pc.sum()), 1e-12)
        eps_sum = resolved = max_gain = 0.0
        violations = 0
        for i, ell in index.cells:
            m_cell = float(pc[i, ell])
            if m_cell < n_pop:
                continue
            V, se_v, src = cell_value(index, i, ell, ev, mode=mode, n_half=n_half, n_pop=n_pop)
            if src in ("exact", "br"):
                g = max(0.0, float(ev.pop_mean[i, ell]) - V)
                se_g = math.sqrt(se_v**2 + float(ev.pop_var[i, ell]) / m_cell)
                eps_sum += m_cell * g
                if m_cell >= n_cap:
                    max_gain = max(max_gain, g)
                    violations += int(g - eps_cell > 2.0 * se_g)
            sl = index.slice(i, ell)
            if br_value(ev.act_n[0][sl], ev.act_mean[0][sl], ev.act_n[1][sl], ev.act_mean[1][sl], n_half) is not None:
                resolved += m_cell
        stats.update(
            {
                "br_eps_regret_per_mile": eps_sum / mass,
                "br_eps_max_cell_regret": max_gain,
                "br_eps_cell_violations": float(violations),
                "br_eps_resolved_joiner_mass": resolved / mass,
            }
        )
    return stats


def certificate_passed(stats: dict, tol: dict, mode: str) -> bool:
    ok = (
        abs(stats["entry_signed_residual"]) <= tol["entry_signed"]
        and stats["entry_mean_abs_residual"] <= tol["entry_mean_abs"]
        and stats["entry_offer_coverage"] >= tol["entry_coverage"]
        and stats["unfinished_share"] <= tol["unfinished_share"]
        and stats["exact_bound_violation_share"] <= tol["exact_bound_violation"]
    )
    if mode == "strategic":
        ok = ok and (
            stats["br_eps_regret_per_mile"] <= tol["eps"]
            and stats["br_eps_cell_violations"] == 0
            and stats["br_eps_resolved_joiner_mass"] >= tol["br_resolved_mass"]
        )
    return bool(ok)


@dataclass
class LearningConfig:
    mode: str = "strategic"  # "strategic" or "myopic"
    clearing: bool = False  # sigma_1 if True; sigma_infinity otherwise.
    seed: int = 42
    max_outer: int = 400
    inner_iters: int = 4  # Weight updates per entry update.
    batch_horizon: float = 360.0  # Minutes.
    batch_burn_in: float = 30.0
    batch_tail: float = 120.0
    probe_share: float = 0.05  # Share of joiners exploring uniformly random threshold groups.
    zeta_init: float = 0.80
    gamma_init_exploration: float = 0.50
    warmup_outer: int = 20  # Iterations before excluding undersampled waiting groups.
    beta0: float = 20.0  # Weight step: beta0 / (1 + t)^beta_exponent.
    beta_exponent: float = 0.50
    gamma_floor: float = 1e-4  # Uniform mixing weight.
    eta0: float = 0.60  # Entry step: max(eta0 / (1 + k)^eta_exponent, eta_min).
    eta_exponent: float = 0.70
    eta_min: float = 0.10
    est_cap: float = 300.0  # Effective threshold sample cap.
    pop_cap: float = 2000.0
    est_decay: float = 0.97
    n_use: int = 10  # Samples required to play a waiting group.
    n_half: int = 10  # Samples per half for best-response estimates.
    n_pop: int = 30  # Riders required for a population estimate.
    n_cap: int = 500  # Riders required for the per-cell regret test.
    screen_every: int = 5
    screen_reps: int = 6
    audit_reps: int = 60
    audit_horizon: float = 2400.0
    audit_burn_in: float = 180.0
    audit_tail: float = 300.0
    polish_step: float = 0.5  # Entry adjustment after a failed audit.
    patience: int = 2  # Consecutive passing audits required.
    workers: int = 2  # Audit processes.
    tol: dict = field(
        default_factory=lambda: {
            "entry_signed": 0.002,  # Absolute mean signed entry error.
            "entry_mean_abs": 0.01,  # Mean absolute entry error.
            "entry_coverage": 0.90,  # Offer share with an estimated value.
            "eps": 0.005,  # Mean regret, dollars per mile.
            "eps_cell": 0.02,  # Per-cell regret, dollars per mile.
            "br_resolved_mass": 0.50,  # Joiner share with a split-sample value.
            "unfinished_share": 0.01,  # Sampled riders still waiting at the horizon.
            "exact_bound_violation": 0.01,
            "screen_entry_signed": 0.004,  # Screening thresholds for a full audit.
            "screen_eps": 0.01,
        }
    )


def train(mech: Mechanism, cfg: LearningConfig, *, log: Callable[[str], None] = print) -> dict:
    index = ActionIndex(mech)
    rng = np.random.default_rng(cfg.seed + 7_919)  # Audit seeds.
    strategic = cfg.mode == "strategic"
    probe_share = cfg.probe_share if strategic else 0.0
    act = RunningStats((2, index.size), cfg.est_cap, cfg.est_decay)
    pop = RunningStats((mech.n, mech.n + 1), cfg.pop_cap, cfg.est_decay)
    sim = LearningSimulator(mech, index, probe_share=probe_share, seed=cfg.seed, clearing=cfg.clearing)
    zeta = initial_zeta(mech, cfg.zeta_init)
    weights = initial_gamma(mech, index, cfg.gamma_init_exploration) if strategic else myopic_gamma(mech)
    gamma, played_n = weights.copy(), np.zeros(index.size)
    history, consecutive, t_step, last_cert = [], 0, 0, {}

    def batch() -> dict:
        out = run_truncated(
            sim, zeta, gamma, mode=cfg.mode, horizon=cfg.batch_horizon, burn_in=cfg.batch_burn_in, tail=cfg.batch_tail
        )
        merge(out)
        return out

    def merge(out: dict) -> None:
        act.update(out["act_sum"], out["act_sq"], out["act_cnt"])
        pop.update(out["pop_sum"], out["pop_sq"], out["pop_cnt"])

    def refresh(k: int) -> None:
        nonlocal gamma, played_n
        played_n = act.n[0] + act.n[1]
        gamma = (
            played_strategy(index, weights, act, cfg.n_use) if strategic and k >= cfg.warmup_outer else weights.copy()
        )

    def weight_step(k: int) -> float:
        nonlocal t_step
        beta = cfg.beta0 / (1.0 + t_step) ** cfg.beta_exponent
        t_step += 1
        delta = mw_update(index, weights, act, pop, beta=beta, n_use=cfg.n_use, floor=cfg.gamma_floor)
        refresh(k)
        return delta

    def audit(reps, horizon, burn_in, tail) -> tuple[dict, dict]:
        agg = run_audit(
            mech,
            index,
            zeta,
            gamma,
            mode=cfg.mode,
            clearing=cfg.clearing,
            probe_share=probe_share,
            seeds=rng.integers(0, 2**62, size=reps),
            horizon=horizon,
            burn_in=burn_in,
            tail=tail,
            workers=cfg.workers,
        )
        stats = certificate_stats(
            index,
            zeta,
            agg,
            mode=cfg.mode,
            n_half=cfg.n_half,
            n_pop=cfg.n_pop,
            n_cap=cfg.n_cap,
            eps_cell=cfg.tol["eps_cell"],
        )
        return agg, stats

    for k in range(cfg.max_outer):
        frozen = consecutive > 0  # Keep the profile fixed between passing audits.
        if not frozen:
            refresh(k)
        wdelta = 0.0
        for _ in range(cfg.inner_iters):
            batch()
            if strategic and not frozen:
                wdelta = max(wdelta, weight_step(k))
        out = batch()
        if not frozen:  # Update entry probabilities.
            target = entry_targets(index, running_evidence(act, pop), mode=cfg.mode, n_half=cfg.n_half, n_pop=cfg.n_pop)
            eta = max(cfg.eta0 / (1.0 + k) ** cfg.eta_exponent, cfg.eta_min)
            ok = np.isfinite(target)
            zeta[ok] = (1.0 - eta) * zeta[ok] + eta * target[ok]
        row = {
            "outer": k,
            "weight_delta": wdelta,
            "frozen": float(frozen),
            "train_profit": out["profit"],
            "train_unfinished": out["unfinished"],
        }

        if frozen or (k + 1) % cfg.screen_every == 0:
            go = True
            if not frozen:
                agg, sstats = audit(cfg.screen_reps, 1800.0, 180.0, 120.0)
                row.update({f"screen_{a}": b for a, b in sstats.items()})
                go = abs(sstats["entry_signed_residual"]) <= cfg.tol["screen_entry_signed"] and (
                    not strategic
                    or (k >= cfg.warmup_outer and sstats["br_eps_regret_per_mile"] <= cfg.tol["screen_eps"])
                )
                merge(agg)
            if go:
                agg, astats = audit(cfg.audit_reps, cfg.audit_horizon, cfg.audit_burn_in, cfg.audit_tail)
                passed = certificate_passed(astats, cfg.tol, cfg.mode)
                row.update({f"audit_{a}": b for a, b in astats.items()})
                row["audit_passed"] = float(passed)
                last_cert = astats
                consecutive = consecutive + 1 if passed else 0
                if not passed:
                    merge(agg)
                    target = entry_targets(
                        index, audit_evidence(agg), mode=cfg.mode, n_half=cfg.n_half, n_pop=cfg.n_pop
                    )
                    ok = np.isfinite(target)
                    zeta[ok] = zeta[ok] + cfg.polish_step * (target[ok] - zeta[ok])
                    if strategic:
                        weight_step(k)
        row["consecutive"] = float(consecutive)
        history.append(row)
        log(_fmt(cfg.mode, row))
        if consecutive >= cfg.patience:
            break
    return {
        "zeta": zeta,
        "gamma": gamma,
        "weights": weights,
        "played_n": played_n,
        "history": history,
        "certified": consecutive >= cfg.patience,
        "certificate": last_cert,
    }


def _fmt(mode: str, row: dict) -> str:
    keys = (
        "outer",
        "weight_delta",
        "screen_entry_signed_residual",
        "screen_br_eps_regret_per_mile",
        "audit_entry_signed_residual",
        "audit_entry_mean_abs_residual",
        "audit_entry_offer_coverage",
        "audit_br_eps_regret_per_mile",
        "audit_br_eps_max_cell_regret",
        "audit_passed",
        "consecutive",
    )
    return " ".join([f"[{mode}]"] + [f"{k}={row[k]:.4g}" for k in keys if k in row])
