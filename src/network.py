# This file computes shortest-path distances on the Chicago road network.

from __future__ import annotations

import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def load_network(path):
    # Edge lengths are in meters.
    with open(path, "rb") as f:
        return pickle.load(f)["G"]


def distance_blocks(G, pickups: np.ndarray, dropoffs: np.ndarray) -> dict[str, np.ndarray]:
    # Distances in meters; P = pickups, Q = drop-offs.
    # L_diag[i] = dist(P_i, Q_i).
    # Blocks [i, j]: PP = P_i->P_j, PQ = P_i->Q_j, QQ = Q_i->Q_j.
    nodes = list(G.nodes())
    pos = {int(u): k for k, u in enumerate(nodes)}
    weight: dict[tuple[int, int], float] = {}
    for u, v, length in G.edges(data="length"):
        length = float(length)
        if not np.isfinite(length) or length < 0:
            raise ValueError(f"invalid edge length {length} on ({u}, {v})")
        key = (pos[int(u)], pos[int(v)])
        weight[key] = min(length, weight.get(key, np.inf))  # Keep the shortest parallel edge.
    rows, cols = np.array(list(weight)).T
    graph = csr_matrix((np.fromiter(weight.values(), float), (rows, cols)), shape=(len(nodes), len(nodes)))

    P, Q = np.asarray(pickups, dtype=int), np.asarray(dropoffs, dtype=int)
    endpoints = list(dict.fromkeys(int(x) for x in np.concatenate([P, Q])))
    dist = dijkstra(graph, directed=True, indices=[pos[x] for x in endpoints])
    row = {x: k for k, x in enumerate(endpoints)}

    def block(A, B):
        return dist[np.ix_([row[int(a)] for a in A], [pos[int(b)] for b in B])]

    out = {"L_diag": block(P, Q).diagonal().copy(), "L_PP": block(P, P), "L_PQ": block(P, Q), "L_QQ": block(Q, Q)}
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError("some type endpoints are not connected in the road network")
    return out
