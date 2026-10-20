#!/usr/bin/env python3
# This file generates Table 1 from the experiment results.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

POLICY_NAMES = {
    "myopic": "Myopic",
    "laissez_faire": r"Strategic, $\sigma_\infty$",
    "clearing": r"Strategic, $\sigma_1$",
}
ALL, STRATEGIC = ("myopic", "laissez_faire", "clearing"), ("laissez_faire", "clearing")
CHANGES = (
    (("myopic", "laissez_faire"), r"Myopic vs. $\sigma_\infty$"),
    (("laissez_faire", "clearing"), r"$\sigma_\infty$ vs. $\sigma_1$"),
)


def _fmt(x: float, digits: int) -> str:
    if not np.isfinite(x):
        return "-"
    s = f"{abs(x):.{digits}f}"
    return s if x >= 0 or float(s) == 0 else f"$-${s}"  # avoid negative zero


def _level(metric: str):
    return lambda row: _fmt(row[metric], 3)


BLOCKS = (  # label, policies, formatter, percentage-change metric
    (r"\shortstack{\textbf{Platform Profit}\\(\$/min)}", ALL, _level("profit"), "profit"),
    (r"\shortstack{\textbf{Rider Surplus}\\(\$/min)}", ALL, _level("rider_surplus"), "rider_surplus"),
    (r"\shortstack{\textbf{Payment}\\(\$/mile$\:\cdot\:$request)}", ALL, _level("payment"), None),
    (r"\shortstack{\textbf{Throughput}\\(requests/min)}", ALL, _level("throughput"), None),
    (r"\textbf{Match Rate}", ALL, _level("match_rate"), None),
    (r"\textbf{Cost Efficiency}", ALL, _level("cost_efficiency"), None),
    (r"\shortstack{\textbf{Actual-Queue}\\ \textbf{Waiting Time} (min)}", ALL, _level("actual_queue_wait"), None),
    (r"\shortstack{\textbf{Shadow-Queue}\\ \textbf{Waiting Time} (min)}", STRATEGIC, _level("shadow_queue_wait"), None),
    (r"\shortstack{\textbf{Fraction of}\\\textbf{Delayed Requests}}", STRATEGIC, _level("share_waited"), None),
)


def _pct(old: float, new: float) -> str:
    if not (np.isfinite(old) and np.isfinite(new)) or abs(old) < 1e-12:
        return "-"
    return _fmt(100.0 * (new - old) / abs(old), 1) + r"\%"


def _short(x: float) -> str:
    return f"{x:g}" if x != int(x) else f"{x:.1f}"


def make_table(df: pd.DataFrame, detour_cap: float) -> str:
    costs, ws = sorted(df["cost"].unique()), sorted(df["w"].unique())
    value = df.set_index(["policy", "cost", "w"])
    missing = [(p, c, w) for p in ALL for c in costs for w in ws if (p, c, w) not in value.index]
    if missing or not df["certified"].all():
        raise ValueError(f"missing results {missing}" if missing else "some profiles are not certified")

    def cells(fmt) -> list[str]:
        lines = ["    & " + " & ".join(fmt(c, w) for w in ws) for c in costs]
        return lines[:-1] + [lines[-1] + r" \\"]

    k, ncol = len(ws), 2 + len(costs) * len(ws)
    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\captionsetup{font=bf}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{cl*{{{ncol - 2}}}{{c}}}}",
        r"\toprule",
        r"\multicolumn{2}{c}{\textbf{Dispatch Cost} (\$/mile)}",
    ]
    out += [rf"    & \multicolumn{{{k}}}{{c}}{{$c = {_short(c)}$}}" for c in costs]
    out[-1] += r" \\"
    out += [rf"\cmidrule(lr){{{3 + g * k}-{2 + (g + 1) * k}}}" for g in range(len(costs))]
    out += [r"\multicolumn{2}{c}{\textbf{Waiting Sensitivity} $w$ (\$/mile$\:\cdot\:$hr)}"]
    out += cells(lambda c, w: _short(w))
    for label, policies, cell, change in BLOCKS:
        changes = [(pair, name) for pair, name in CHANGES if change and set(pair) <= set(policies)]
        out += [r"\midrule", "", rf"\multirow{{{len(policies) + len(changes)}}}{{*}}{{{label}}}"]
        for policy in policies:
            out.append(f"    & {POLICY_NAMES[policy]}")
            out += cells(lambda c, w, p=policy, f=cell: f(value.loc[(p, c, w)]))
        if changes:
            out.append(r"\addlinespace[1pt]")
            for (old, new), name in changes:
                out.append(f"    & {name}")
                out += cells(
                    lambda c, w, o=old, n=new, m=change: _pct(value.loc[(o, c, w), m], value.loc[(n, c, w), m])
                )
    reps = int(df["replications"].iloc[0])
    out += [
        r"\bottomrule",
        "",
        r"\end{tabular}",
        "}",
        r"\vspace{-0.5em}",
        rf"\caption{{Numerical results on Chicago ridesharing data ($\bar{{\delta}} = {round(100 * detour_cap)}\%$, "
        rf"$\theta = 1$), averaged over {reps} simulation runs.}}",
        r"\label{tab:chicago-results}",
        r"\end{table}",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    out = ROOT / "results" / "table1.tex"
    out.write_text(make_table(pd.read_csv(ROOT / "results" / "table1_metrics.csv"), detour_cap=0.25), newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
