# This file simulates rider arrivals, matching, and departures.

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .instance import Instance

TOL = 1e-12
TYPE_STATS = (
    "arrivals",
    "joined",
    "delayed",
    "discount",
    "completed",
    "matched",
    "wait",
    "shadow_wait",
    "waited",
    "queued",
    "payment_per_mile",
    "surplus",
)


@dataclass
class Mechanism:
    # Price labels: 0 = solo; j + 1 = match with type j.

    payments: np.ndarray  # (n, n + 1); NaN for incompatible pairs
    solo_miles: np.ndarray  # d_i
    vehicle_miles: np.ndarray  # d_ij
    arrival_rates: np.ndarray  # riders per minute
    compatible: np.ndarray  # bool (n, n)
    reward: np.ndarray  # p_ij + p_ji - c d_ij; NaN if incompatible
    cost_per_mile: float  # c
    theta: float  # window expiry rate per minute
    waiting_cost: float  # dollars per solo mile per minute (w / 60)
    labels: list = field(init=False)  # available price labels
    partners: list = field(init=False)  # compatible, nonnegative-profit partners, including i
    offer_order: list = field(init=False)  # other partners by profit; ties favor lower type
    watchers: list = field(init=False)  # types affected by actual-queue slot k

    def __post_init__(self) -> None:
        n = self.n
        ok = self.compatible & (np.nan_to_num(self.reward, nan=-np.inf) >= -TOL)
        self.labels = [np.concatenate([[0], np.flatnonzero(self.compatible[i]) + 1]).astype(int) for i in range(n)]
        self.partners = [[int(j) for j in np.flatnonzero(ok[i])] for i in range(n)]
        self.offer_order = [
            sorted((j for j in self.partners[i] if j != i), key=lambda j, i=i: (-self.reward[i, j], j))
            for i in range(n)
        ]
        self.watchers = [[k] for k in range(n)]
        for i in range(n):
            for k in self.offer_order[i]:
                self.watchers[k].append(i)
        if not all(ok[i, i] and self.payments[i, i + 1] <= np.nanmin(self.payments[i, 1:]) + TOL for i in range(n)):
            raise ValueError("the same-type match must be the cheapest offer and profitable")

    @property
    def n(self) -> int:
        return len(self.solo_miles)


def make_mechanism(inst: Instance, *, theta: float, waiting_cost: float) -> Mechanism:
    P, K, c = inst.payments, inst.compatible.astype(bool), inst.dispatch_cost
    reward = np.where(K, P[:, 1:] + P[:, 1:].T - c * inst.vehicle_miles, np.nan)
    return Mechanism(
        payments=P,
        solo_miles=inst.solo_miles,
        vehicle_miles=inst.vehicle_miles,
        arrival_rates=inst.arrival_rates,
        compatible=K,
        reward=reward,
        cost_per_mile=c,
        theta=float(theta),
        waiting_cost=float(waiting_cost),
    )


@dataclass(slots=True)
class Rider:
    rider_id: int
    typ: int
    threshold: int  # price label q
    max_price: float  # highest accepted price
    offer: int  # offer label on arrival
    join_time: float
    probe: bool = False  # uniform threshold exploration
    shadow_since: float | None = None
    shadow_wait: float = 0.0
    queued_since: float | None = None  # actual-queue entry time
    seq: int = 0  # shadow-queue entry order


class Simulator:

    def __init__(self, mech: Mechanism, *, seed: int, clearing: bool = False) -> None:
        self.mech = mech
        self.rng = np.random.default_rng(seed)
        self.clearing = bool(clearing)
        self.next_id = 0
        self.price = mech.payments.tolist()
        self.reward = mech.reward.tolist()
        self.miles = mech.solo_miles.tolist()
        self.route = mech.vehicle_miles.tolist()
        self.pair_ok = [set(p) for p in mech.partners]

    def new_rider(self, typ: int, threshold: int, offer: int, now: float) -> Rider:
        self.next_id += 1
        return Rider(self.next_id, typ, threshold, self.price[typ][threshold], offer, now)

    def on_offer(self, typ: int, offer: int) -> None:
        pass

    def on_match(self, a: Rider, b: Rider, now: float) -> None:
        pass

    def on_departure(self, ag: Rider, payment: float, now: float) -> None:
        pass

    def phi(self, i: int) -> int:
        actual = self.actual
        if actual[i] is not None:
            return i + 1
        for k in self.mech.offer_order[i]:
            if actual[k] is not None:
                return k + 1
        return 0

    def _accepts(self, ag: Rider, label: int) -> bool:
        return self.price[ag.typ][label] <= ag.max_price + TOL

    def _complete(self, ag: Rider, payment: float, matched: bool, now: float) -> None:
        if ag.shadow_since is not None:
            ag.shadow_wait += now - ag.shadow_since
            ag.shadow_since = None
        if now < self.burn_in:
            return
        i, s, ts = ag.typ, self.stats, self.type_stats
        d = self.miles[i]
        wait, per_mile, waited = now - ag.join_time, payment / d, float(ag.shadow_wait > 0)
        queued = float(ag.queued_since is not None and now > ag.queued_since)
        # Uniform valuations: E[value | join] = d * (1 - zeta / 2).
        surplus = d * (1.0 - self.join[i][ag.offer] / 2.0) - payment - self.mech.waiting_cost * d * wait
        s["completed"] += 1.0
        s["matched"] += float(matched)
        s["wait"] += wait
        s["shadow_wait"] += ag.shadow_wait
        s["waited"] += waited
        s["payment_per_mile"] += per_mile
        s["solo_miles"] += d
        s["queued"] += queued
        s["surplus"] += surplus
        for key, value in (
            ("completed", 1.0),
            ("matched", float(matched)),
            ("wait", wait),
            ("shadow_wait", ag.shadow_wait),
            ("waited", waited),
            ("payment_per_mile", per_mile),
            ("queued", queued),
            ("surplus", surplus),
        ):
            ts[key][i] += value

    def _serve_solo(self, i: int, now: float) -> None:
        ag = self._take(i)
        payment = self.price[i][0]
        self.on_departure(ag, payment, now)
        if now >= self.burn_in:
            self.stats["profit"] += payment - self.mech.cost_per_mile * self.miles[i]
            self.stats["vehicle_miles"] += self.miles[i]
        self._complete(ag, payment, False, now)

    def _serve_match(self, a: Rider, b: Rider, now: float) -> None:
        i, j = a.typ, b.typ
        pay_a, pay_b = self.price[i][j + 1], self.price[j][i + 1]
        if not (self._accepts(a, j + 1) and self._accepts(b, i + 1)):
            raise RuntimeError("a rider was matched at a price above her threshold")
        self.on_match(a, b, now)
        self.on_departure(a, pay_a, now)
        self.on_departure(b, pay_b, now)
        if now >= self.burn_in:
            self.stats["profit"] += pay_a + pay_b - self.mech.cost_per_mile * self.route[i][j]
            self.stats["vehicle_miles"] += self.route[i][j]
        self._complete(a, pay_a, True, now)
        self._complete(b, pay_b, True, now)

    def _take(self, k: int) -> Rider:
        ag, self.actual[k] = self.actual[k], None
        self.occupied.discard(k)
        return ag

    def _admit(self, ag: Rider, now: float) -> None:
        ag.queued_since = now
        self.actual[ag.typ] = ag
        self.occupied.add(ag.typ)

    def _request(self, ag: Rider, now: float) -> int:
        # Return the actual-queue slot changed by the request.
        ell = self.phi(ag.typ)
        if ell > 0:
            self._serve_match(ag, self._take(ell - 1), now)
            return ell - 1
        if self.actual[ag.typ] is None and self._accepts(ag, 0):
            self._admit(ag, now)
            return ag.typ
        raise RuntimeError("an immediate request could not be served")

    def _to_shadow(self, ag: Rider) -> None:
        self.seq += 1
        ag.seq = self.seq
        self.shadow.setdefault(ag.typ, []).append(ag)

    def _best_pair(self, requesters: list) -> tuple | None:
        # Maximize profit; the incumbent, if any, comes first.
        actual, price, reward, best = self.actual, self.price, self.reward, None
        for r_pos, r in enumerate(requesters):
            i = r.typ
            for j in self.mech.partners[i]:
                inc = actual[j]
                if inc is not None and price[i][j + 1] <= r.max_price + TOL and price[j][i + 1] <= inc.max_price + TOL:
                    key = (-reward[i][j], 0, j, r_pos)
                    if best is None or key < best[0]:
                        best = (key, inc, r)
            for s_pos in range(r_pos + 1, len(requesters)):
                s = requesters[s_pos]
                k = s.typ
                if (
                    k in self.pair_ok[i]
                    and price[i][k + 1] <= r.max_price + TOL
                    and price[k][i + 1] <= s.max_price + TOL
                ):
                    key = (-reward[i][k], 1, r_pos, s_pos)
                    if best is None or key < best[0]:
                        best = (key, r, s)
        return best

    def _batch(self, now: float, slot: int) -> None:
        # Match triggered shadow riders by profit, cascading after queue changes.
        requesters: list[Rider] = []
        changed: int | None = slot
        while True:
            if changed is not None:
                triggered = []
                for i in self.mech.watchers[changed]:
                    riders = self.shadow.get(i)
                    if not riders:
                        continue
                    p = self.price[i][self.phi(i)]
                    stay = [ag for ag in riders if not p <= ag.max_price + TOL]
                    if len(stay) < len(riders):
                        triggered += [ag for ag in riders if p <= ag.max_price + TOL]
                        self.shadow[i] = stay
                triggered.sort(key=lambda ag: ag.seq)
                requesters += triggered
            best = self._best_pair(requesters) if requesters else None
            if best is None:
                for ag in requesters:
                    self._to_shadow(ag)
                return
            _, a, b = best
            changed = a.typ if self.actual[a.typ] is a else None
            if changed is not None:
                self._take(changed)
            self._serve_match(a, b, now)
            requesters = [ag for ag in requesters if ag is not a and ag is not b]

    def _draw_threshold(self, i: int, ell: int) -> int:
        # Use the same random draw as Generator.choice.
        cached = self._cdf.get((i, ell))
        if cached is None:
            labels = self.mech.labels[i]
            probs = self.gamma[i, labels, ell]
            s = probs.sum()
            cdf = None
            if np.isfinite(s) and s > 0:
                cdf = (probs / s).cumsum()
                cdf = (cdf / cdf[-1]).tolist()
            cached = self._cdf[(i, ell)] = (labels.tolist(), cdf)
        labels, cdf = cached
        if cdf is None:
            return labels[int(self.rng.integers(0, len(labels)))]
        return labels[bisect_right(cdf, self.rng.random())]

    def run(self, zeta: np.ndarray, gamma: np.ndarray, *, mode: str, horizon: float, burn_in: float) -> dict:
        # zeta[i, ell]: join probability; gamma[i, q, ell]: threshold probability.
        if mode not in ("strategic", "myopic"):
            raise ValueError(mode)
        m, rng, n = self.mech, self.rng, self.mech.n
        strategic = mode == "strategic"
        self.burn_in = float(burn_in)
        self.stats = dict.fromkeys(
            (
                "profit",
                "completed",
                "matched",
                "wait",
                "shadow_wait",
                "waited",
                "payment_per_mile",
                "solo_miles",
                "vehicle_miles",
                "arrivals",
                "joined",
                "delayed",
                "discount",
                "queued",
                "surplus",
            ),
            0.0,
        )
        self.type_stats = {k: np.zeros(n) for k in TYPE_STATS}
        self.actual: list[Rider | None] = [None] * n
        self.occupied: set[int] = set()
        self.shadow: dict[int, list[Rider]] = {}
        self.seq = 0
        self.gamma, self._cdf = gamma, {}
        self.join = join = zeta.tolist()
        total_rate = float(m.arrival_rates.sum())
        cdf = m.arrival_rates / total_rate
        cdf = cdf.cumsum()
        cdf = (cdf / cdf[-1]).tolist()
        t = 0.0
        while t < horizon:
            event_rate = total_rate + m.theta * len(self.occupied)
            t += float(rng.exponential(1.0 / event_rate))
            if t > horizon:
                break
            if rng.random() >= total_rate / event_rate:  # window expiry
                occupied = sorted(self.occupied)
                k = occupied[int(rng.integers(0, len(occupied)))]
                self._serve_solo(k, t)
                if strategic:
                    self._batch(t, k)
                continue
            i = bisect_right(cdf, rng.random())
            ell = self.phi(i)
            counted = t >= self.burn_in
            if counted:
                self.stats["arrivals"] += 1.0
                self.type_stats["arrivals"][i] += 1.0
                self.on_offer(i, ell)
            if rng.random() > join[i][ell]:
                continue
            if counted:
                self.stats["joined"] += 1.0
                self.type_stats["joined"][i] += 1.0
            ag = self.new_rider(i, self._draw_threshold(i, ell) if strategic else 0, ell, t)
            if not strategic or self._accepts(ag, ell):
                slot = self._request(ag, t)
                if strategic:
                    self._batch(t, slot)
                continue
            ag.shadow_since = t
            if counted:
                discount = (self.price[i][0] - ag.max_price) / self.price[i][0]
                self.stats["delayed"] += 1.0
                self.stats["discount"] += discount
                self.type_stats["delayed"][i] += 1.0
                self.type_stats["discount"][i] += discount
            waiting = self.shadow.get(i)
            # Clearing matches the oldest shadow rider of the same type.
            if self.clearing and waiting:
                self._serve_match(ag, waiting.pop(0), t)
            else:
                self._to_shadow(ag)
        shadow = sorted((ag for riders in self.shadow.values() for ag in riders), key=lambda ag: ag.seq)
        return {"stats": self.stats, "type_stats": self.type_stats, "actual": self.actual, "shadow": shadow}


def summarize(stats: dict, duration: float) -> dict:
    # duration = horizon - burn_in, in minutes.
    done = stats["completed"]
    per = (lambda x: x / done) if done > 0 else (lambda x: float("nan"))
    return {
        "profit": stats["profit"] / duration,
        "payment": per(stats["payment_per_mile"]),
        "throughput": done / duration,
        "match_rate": per(stats["matched"]),
        "cost_efficiency": (
            (stats["solo_miles"] - stats["vehicle_miles"]) / stats["solo_miles"]
            if stats["solo_miles"] > 0
            else float("nan")
        ),
        "actual_queue_wait": per(stats["wait"] - stats["shadow_wait"]),
        "shadow_queue_wait": per(stats["shadow_wait"]),
        "share_waited": per(stats["waited"]),
        "wait_if_waited": stats["shadow_wait"] / stats["waited"] if stats["waited"] > 0 else float("nan"),
        "join_rate": stats["joined"] / stats["arrivals"] if stats["arrivals"] > 0 else float("nan"),
        "delay_rate": stats["delayed"] / stats["joined"] if stats["joined"] > 0 else float("nan"),
        "rider_surplus": stats["surplus"] / duration,
        "share_queued": per(stats["queued"]),
        "wait_if_queued": (
            (stats["wait"] - stats["shadow_wait"]) / stats["queued"] if stats["queued"] > 0 else float("nan")
        ),
        "request_discount": stats["discount"] / stats["joined"] if stats["joined"] > 0 else float("nan"),
    }


_EVAL: dict = {}  # evaluation state shared with workers


def _init_eval(state: dict) -> None:
    _EVAL.clear()
    _EVAL.update(state)


def _replication(r: int) -> tuple[dict, dict]:
    c = _EVAL
    sim = Simulator(c["mech"], seed=c["seed"] + 1009 * r, clearing=c["clearing"])
    out = sim.run(c["zeta"], c["gamma"], mode=c["mode"], horizon=c["horizon"], burn_in=c["burn_in"])
    return {"replication": r, **summarize(out["stats"], c["horizon"] - c["burn_in"])}, out["type_stats"]


def evaluate(
    mech: Mechanism,
    zeta: np.ndarray,
    gamma: np.ndarray,
    *,
    mode: str,
    clearing: bool,
    seed: int,
    replications: int,
    horizon: float = 1800.0,
    burn_in: float = 180.0,
    workers: int = 1,
):
    state = dict(
        mech=mech,
        zeta=zeta,
        gamma=gamma,
        mode=mode,
        clearing=clearing,
        seed=int(seed),
        horizon=float(horizon),
        burn_in=float(burn_in),
    )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_eval, initargs=(state,)) as ex:
            outs = list(ex.map(_replication, range(replications)))
    else:
        _init_eval(state)
        outs = [_replication(r) for r in range(replications)]
    type_totals = {k: sum(o[1][k] for o in outs) for k in TYPE_STATS}
    return [o[0] for o in outs], type_totals
