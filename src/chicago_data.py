# This file prepares Chicago trip data and constructs rider types.

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

WINDOW = ("07:30", "08:30")  # Morning peak.


@dataclass(slots=True)
class RiderTypes:
    centers: pd.DataFrame  # Pickup/drop-off locations and arrival rate, one row per type.
    arrival_rates: np.ndarray  # riders per minute
    service_dates: list[str]


def load_trips(path: str) -> pd.DataFrame:
    # Shared-authorized peak trips; coordinates in UTM meters.
    df = pd.read_csv(path)
    for col in ("pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "pickup_area"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["Trip Start Timestamp"], errors="coerce")
    df = df.dropna(subset=["pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "pickup_area", "timestamp"])
    df["pickup_area"] = df["pickup_area"].astype(int)
    df["service_date"] = df["timestamp"].dt.date.astype(str)
    shared = df["Shared Trip Authorized"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    tod = df["timestamp"].dt.time
    start, end = (pd.to_datetime(t).time() for t in WINDOW)
    return df.loc[shared & (tod >= start) & (tod < end)].copy()


def window_minutes() -> float:
    start, end = (pd.to_datetime(t) for t in WINDOW)
    return (end - start).total_seconds() / 60.0


def _load_zones(path: str) -> tuple[dict[int, int], pd.DataFrame]:
    # Community-to-zone mapping and zone centroids.
    with open(path, "rb") as f:
        z = pickle.load(f)
    areas = z["shapes_gdf"][["area_num_1", "cluster"]].astype(int)
    zones = z["clusters_gdf"][["cluster", "x", "y", "cluster_centroid_node"]].copy()
    zones[["cluster", "cluster_centroid_node"]] = zones[["cluster", "cluster_centroid_node"]].astype(int)
    return dict(zip(areas["area_num_1"], areas["cluster"], strict=True)), zones.set_index("cluster").sort_index()


def _clusters_per_zone(zone_counts: pd.Series, distinct_dropoffs: pd.Series, n_types: int) -> pd.Series:
    # Allocate clusters proportionally, subject to zone capacities.
    upper = np.minimum(distinct_dropoffs.reindex(zone_counts.index).to_numpy(int), zone_counts.to_numpy(int))
    if n_types < len(zone_counts) or upper.sum() < n_types:
        raise ValueError(f"cannot form {n_types} types from {len(zone_counts)} zones")
    quota = n_types * zone_counts.to_numpy(float) / zone_counts.sum()
    alloc = np.minimum(np.maximum(1, np.floor(quota).astype(int)), upper)
    while alloc.sum() < n_types:
        cand = np.flatnonzero(alloc < upper)
        alloc[cand[np.argmax(quota[cand] - alloc[cand])]] += 1
    while alloc.sum() > n_types:
        cand = np.flatnonzero(alloc > 1)
        alloc[cand[np.argmax(alloc[cand] - quota[cand])]] -= 1
    return pd.Series(alloc, index=zone_counts.index)


def _nearest_nodes(G, xs, ys) -> np.ndarray:
    ids, coords = zip(*[(int(u), (float(d["x"]), float(d["y"]))) for u, d in G.nodes(data=True)], strict=True)
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(coords))
    _, idx = nn.kneighbors(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]))
    return np.asarray(ids)[idx[:, 0]]


def build_rider_types(trips: pd.DataFrame, G, zone_path: str, *, n_types: int, seed: int = 0) -> RiderTypes:
    area_to_zone, zones = _load_zones(zone_path)
    zone_of_trip = trips["pickup_area"].map(area_to_zone)
    if zone_of_trip.isna().any() or set(zone_of_trip.astype(int)) != set(zones.index):
        raise ValueError("every trip must map to a pickup zone, and every zone must have trips")
    zone_of_trip = zone_of_trip.astype(int).to_numpy()
    dropoff = trips[["dropoff_x", "dropoff_y"]].to_numpy(float)

    zone_counts = pd.Series(zone_of_trip).value_counts().sort_index()
    distinct = pd.Series({z: len(np.unique(dropoff[zone_of_trip == z], axis=0)) for z in zone_counts.index})
    k_per_zone = _clusters_per_zone(zone_counts, distinct, n_types)

    rows = []
    for zone in zone_counts.index:
        pts = dropoff[zone_of_trip == zone]
        k = int(k_per_zone[zone])
        km = KMeans(n_clusters=k, random_state=seed + 1009 * int(zone), n_init=10, max_iter=300, algorithm="lloyd").fit(
            pts
        )
        # Order types by centroid position.
        order = np.lexsort((km.cluster_centers_[:, 1], km.cluster_centers_[:, 0]))
        sizes = np.bincount(km.labels_, minlength=k)
        if np.any(sizes == 0):
            raise RuntimeError(f"empty drop-off cluster in zone {zone}")
        for c in order:
            rows.append(
                {
                    "type": len(rows),
                    "pickup_zone": int(zone),
                    "cluster_pickup_node": int(zones.loc[zone, "cluster_centroid_node"]),
                    "cluster_dropoff_x": float(km.cluster_centers_[c, 0]),
                    "cluster_dropoff_y": float(km.cluster_centers_[c, 1]),
                    "cluster_size": int(sizes[c]),
                }
            )
    centers = pd.DataFrame(rows)
    centers["cluster_dropoff_node"] = _nearest_nodes(G, centers["cluster_dropoff_x"], centers["cluster_dropoff_y"])
    dates = sorted(trips["service_date"].unique())
    rates = centers["cluster_size"].to_numpy(float) / (len(dates) * window_minutes())
    centers["arrival_rate_per_min"] = rates
    return RiderTypes(centers=centers, arrival_rates=rates, service_dates=dates)
