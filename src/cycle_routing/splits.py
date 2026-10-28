"""Open/closed zones for the hackathon: where teams see labels and where the organizer grades.

A split gives every road a zone (``open``, ``closed`` or ``buffer``) and OD pairs inside each zone.
Both strategies share the same candidate corridors (shortest path plus alternatives within the detour
limit), so each OD has a real choice and its routes stay inside its own zone:

* ``by_routes``: zones are cut around the corridors by road distance (network Voronoi); a corridor of
  one zone is kept at least ``separation_m`` away from every corridor of the other zone;
* ``by_sectors``: a fixed checkerboard of square sectors; only corridors inside one colour are used.

Only *whether* a road is measured is used here, never the number of trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BANDS = {"1.5–3 км": (1500, 3000), "3–4.5 км": (3000, 4500), "4.5–6 км": (4500, 6000)}
ZONES = ("closed", "open")


@dataclass
class Split:
    zone: pd.Series            # per road: open / closed / buffer
    pairs: pd.DataFrame        # od_id, origin, destination, zone, band, length_km, start/end lat/lon
    params: dict = field(default_factory=dict)


def _simple_graph(edges, directed=True):
    best = edges.sort_values("length_m").drop_duplicates(["u", "v"])
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_weighted_edges_from(zip(best.u, best.v, best.length_m), weight="w")
    return graph, dict(zip(zip(best.u, best.v), best.index))


def _city_mask(edges, measured, cell=1000, min_km=25):
    """Roads in 1 km cells with at least ``min_km`` of measured streets: the dense town."""
    c = edges.geometry.centroid
    cells = pd.MultiIndex.from_arrays([(c.x // cell).astype(int), (c.y // cell).astype(int)])
    km = pd.Series(edges.length_m.where(measured, 0).to_numpy(), index=cells).groupby(level=[0, 1]).sum()
    return pd.Series(km.reindex(cells).to_numpy() >= min_km * 1000, index=edges.index)


def _anchors(edges, nodes, measured, rng, cell=400):
    """One measured intersection per cell, so OD endpoints spread over the whole area."""
    degree = pd.Series(dict(nx.Graph(zip(edges.u, edges.v)).degree()))
    touched = set(edges.u[measured]) | set(edges.v[measured])
    n = nodes.set_index("node").loc[degree.index[degree.ge(3)]]
    n = n[n.index.isin(touched)]
    n = n.assign(cx=n.x // cell, cy=n.y // cell, r=rng.random(len(n)))
    return n.sort_values("r").groupby(["cx", "cy"]).head(1)


def _candidate_pairs(graph, points, rng, per_band):
    ids, rows = set(points.index), []
    top = max(hi for _, hi in BANDS.values())
    for a in points.index:
        for b, d in nx.single_source_dijkstra_path_length(graph, a, cutoff=top, weight="w").items():
            if b in ids and b != a:
                rows.append((a, b, d))
    pairs = pd.DataFrame(rows, columns=["origin", "destination", "sp_m"])
    edges_ = [lo for lo, _ in BANDS.values()] + [top]
    pairs["band"] = pd.cut(pairs.sp_m, edges_, labels=list(BANDS), right=False)
    pairs = pairs.dropna(subset="band").sample(frac=1, random_state=int(rng.integers(1e9)))
    return pairs.groupby("band", observed=True).head(per_band).reset_index(drop=True)


def _corridor(graph, edge_of, length, measured, city, a, b, max_detour, n_alt=2, penalty=3.0):
    """Shortest path plus penalty-method alternatives that stay within the detour limit."""
    sp_len, sp = nx.bidirectional_dijkstra(graph, a, b, weight="w")
    paths, used = [sp], {(x, y) for x, y in zip(sp, sp[1:])}
    for _ in range(n_alt):
        _, alt = nx.bidirectional_dijkstra(graph, a, b, weight=lambda u, v, d: d["w"] * (penalty if (u, v) in used else 1))
        if alt in paths or sum(graph[x][y]["w"] for x, y in zip(alt, alt[1:])) > (1 + max_detour) * sp_len:
            break
        paths.append(alt)
        used |= {(x, y) for x, y in zip(alt, alt[1:])}
    sp_edges = [edge_of[(x, y)] for x, y in zip(sp, sp[1:])]
    sp_length = length.loc[sp_edges].to_numpy()
    shared = np.nan
    if len(paths) > 1:
        alt_pairs = {frozenset(p) for p in zip(paths[1], paths[1][1:])}
        shared = sp_length[[frozenset(p) in alt_pairs for p in zip(sp, sp[1:])]].sum() / sp_len
    return {
        "origin": a, "destination": b, "sp_m": sp_len,
        "measured": sp_length[measured.loc[sp_edges].to_numpy()].sum() / sp_len,
        "city": sp_length[city.loc[sp_edges].to_numpy()].sum() / sp_len,
        "alt_shared": shared,
        "nodes": frozenset().union(*paths),
        "edges": tuple(edge_of[(x, y)] for p in paths for x, y in zip(p, p[1:])),
        "sp_pairs": frozenset(frozenset(p) for p in zip(sp, sp[1:])),
    }


def probe_solutions():
    """Ready presets plus contrasting single-feature weights: a spread of reasonable configurations."""
    from .presets import COMFORT_PRESETS
    from .workshop import comfort_cost
    weights = {"bike": {"велодорожка": -0.7}, "major": {"крупная_дорога": 1.5}, "rough": {"плохое_покрытие": 1.0},
               "quiet": {"жилая_улица": -0.4, "крупная_дорога": 0.8}, "paths": {"пешеходная": -0.5}}
    return {**COMFORT_PRESETS, **{f"probe_{name}": {"cost": comfort_cost(w), "turns": None, "search": "dijkstra"}
                                  for name, w in weights.items()}}


def route_variety(edges, nodes, cands, *, max_detour_pct=20):
    """Different routes the probe solutions give for each candidate within the detour limit, shortest
    included. High variety means a change of configuration is visible on this pair. Uses OSM tags only."""
    from .routing import build_routes
    pairs = [(str(i), int(o), int(d)) for i, o, d in zip(cands.index, cands.origin, cands.destination)]
    routes = build_routes(edges, nodes, pairs, probe_solutions())
    shortest = routes[routes.algorithm.eq("shortest")].set_index("od_id").route_length_m
    near = routes[routes.route_length_m.le((1 + max_detour_pct / 100) * routes.od_id.map(shortest))]
    variety = near.assign(path=near.edge_route.map(lambda p: tuple(map(tuple, p)))).groupby("od_id").path.nunique()
    return pd.Series(variety.reindex([p[0] for p in pairs]).fillna(1).to_numpy(), index=cands.index, name="variety")


def candidate_corridors(edges, nodes, measured, *, seed=7, per_band=600, max_detour_pct=20,
                        min_measured=0.75, max_alt_shared=0.6, min_variety=5, cache: Path | None = None):
    """OD corridors with mostly measured shortest paths, a real alternative and visible differences
    between configurations (``min_variety``). Slow: cache it."""
    if cache is not None and Path(cache).exists():
        cands = pd.read_pickle(cache)
    else:
        rng = np.random.default_rng(seed)
        graph, edge_of = _simple_graph(edges)
        city = _city_mask(edges, measured)
        pairs = _candidate_pairs(graph, _anchors(edges, nodes, measured, rng), rng, per_band)
        cands = pd.DataFrame([
            _corridor(graph, edge_of, edges.length_m, measured, city, p.origin, p.destination, max_detour_pct / 100)
            for p in pairs.itertuples()
        ])
        cands["band"] = pairs["band"].astype(str).to_numpy()
        cands["is_city"] = cands.city.ge(0.5)
        cands = cands[cands.measured.ge(min_measured) & cands.alt_shared.le(max_alt_shared)].reset_index(drop=True)
    if "variety" not in cands:
        cands["variety"] = route_variety(edges, nodes, cands, max_detour_pct=max_detour_pct)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        cands.to_pickle(cache)
    return cands[cands.variety.ge(min_variety)].reset_index(drop=True)


def _pick(cands, xy, *, per_zone_stratum, separation_m, min_endpoint_m, max_overlap, order_seed):
    """Strata x zones are filled in pairs (one closed + one open), so both zones stay equal per stratum.
    Pairs where configurations differ most (``variety``) are tried first."""
    order = cands.sample(frac=1, random_state=order_seed).sort_values("variety", ascending=False, kind="stable").index
    points = {i: np.array([xy[n] for n in cands.at[i, "nodes"]]) for i in order}
    queues = {k: [i for i in order if cands.at[i, "band"] == k[0] and cands.at[i, "is_city"] == k[1]]
              for k in cands.groupby(["band", "is_city"]).groups}
    chosen: list[tuple[int, str]] = []

    def trees():
        return {z: cKDTree(np.vstack([points[i] for i, zz in chosen if zz == z]))
                for z in ZONES if any(zz == z for _, zz in chosen)}

    def usable(i):
        ends = [xy[cands.at[i, "origin"]], xy[cands.at[i, "destination"]]]
        mine = cands.at[i, "sp_pairs"]
        for j, _ in chosen:
            if any(np.linalg.norm(p - xy[cands.at[j, e]]) < min_endpoint_m for p in ends for e in ("origin", "destination")):
                return False
            other = cands.at[j, "sp_pairs"]
            if len(mine & other) > max_overlap * min(len(mine), len(other)):
                return False
        return True

    def fits(i, zone, built):
        fixed = cands.at[i, "fixed"] if "fixed" in cands else None
        if fixed is not None and fixed != zone:
            return False
        other = built.get("open" if zone == "closed" else "closed")
        return other is None or not np.isfinite(other.query(points[i], distance_upper_bound=separation_m)[0]).any()

    active = {k: per_zone_stratum for k in queues}
    while active:
        for k in list(active):
            picked = []
            for zone in ZONES:
                built, found = trees(), None
                for i in list(queues[k]):
                    if not usable(i):
                        queues[k].remove(i)
                    elif fits(i, zone, built):
                        found = i
                        break
                if found is None:
                    break
                queues[k].remove(found)
                chosen.append((found, zone))
                picked.append(found)
            if len(picked) < len(ZONES):
                chosen[:] = [c for c in chosen if c[0] not in picked]
                del active[k]
            else:
                active[k] -= 1
                if not active[k]:
                    del active[k]
    corridors = cands.loc[[i for i, _ in chosen]].assign(zone=[z for _, z in chosen]).reset_index(drop=True)
    gap = np.inf
    if set(corridors.zone) == set(ZONES):
        sides = {z: np.vstack([points[i] for i, zz in chosen if zz == z]) for z in ZONES}
        gap = float(cKDTree(sides["open"]).query(sides["closed"])[0].min())
    return corridors, gap


def _pairs_frame(corridors, nodes, crs):
    xy = nodes.set_index("node")[["x", "y"]]
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    rows = []
    for zone, prefix in (("open", "OD"), ("closed", "CL")):
        part = corridors[corridors.zone.eq(zone)].sort_values(["band", "sp_m"])
        for n, c in enumerate(part.itertuples(), start=1):
            (slon, slat), (elon, elat) = (to_wgs.transform(*xy.loc[c.origin]), to_wgs.transform(*xy.loc[c.destination]))
            rows.append({"od_id": f"{prefix}{n:02d}", "origin": int(c.origin), "destination": int(c.destination),
                         "zone": zone, "band": c.band, "length_km": round(c.sp_m / 1000, 2),
                         "start_lat": slat, "start_lon": slon, "end_lat": elat, "end_lon": elon,
                         "start_label": f"{prefix}{n:02d} A", "end_label": f"{prefix}{n:02d} B"})
    return pd.DataFrame(rows)


def _buffer(graph, closed_nodes, edges, buffer_m):
    near = nx.multi_source_dijkstra_path_length(graph, closed_nodes, cutoff=buffer_m, weight="w") if closed_nodes else {}
    closed = edges.u.isin(closed_nodes) & edges.v.isin(closed_nodes)
    touches = edges.u.isin(near.keys()) | edges.v.isin(near.keys())
    return pd.Series(np.where(closed, "closed", np.where(touches, "buffer", "open")), index=edges.index, name="zone")


def _best(cands, nodes, order_seeds, **pick):
    xy = nodes.set_index("node")[["x", "y"]]
    xy = dict(zip(xy.index, xy.to_numpy()))
    best, best_key = None, None
    for seed in order_seeds:
        corridors, gap = _pick(cands, xy, order_seed=seed, **pick)
        counts = corridors.zone.value_counts()
        city = corridors.groupby("zone").is_city.mean()
        key = (int(counts.min()) if len(counts) == 2 else 0, -abs(city.get("open", 0) - city.get("closed", 0)))
        if best_key is None or key > best_key:
            best, best_key = (corridors, gap, seed), key
    return best


def by_routes(edges, nodes, measured, *, rideable, per_zone_stratum=8, separation_m=800, reach_m=1000,
              buffer_m=150, min_endpoint_m=400, max_overlap=0.25, min_variety=5, order_seeds=range(8), cache=None,
              **candidates):
    """Zones cut around OD corridors: closed = roads nearer (by road) to a closed corridor and within
    ``reach_m`` of it; a ``buffer_m`` strip around it has no labels; all other roads are open."""
    cands = candidate_corridors(edges[rideable], nodes, measured[rideable], min_variety=min_variety, cache=cache, **candidates)
    corridors, gap, seed = _best(cands, nodes, order_seeds, per_zone_stratum=per_zone_stratum,
                                 separation_m=separation_m, min_endpoint_m=min_endpoint_m, max_overlap=max_overlap)
    graph, _ = _simple_graph(edges, directed=False)
    sources = {z: set().union(*corridors.nodes[corridors.zone.eq(z)]) for z in ZONES}
    d_open = nx.multi_source_dijkstra_path_length(graph, sources["open"], weight="w")
    d_closed = nx.multi_source_dijkstra_path_length(graph, sources["closed"], weight="w")
    closed_nodes = {n for n, d in d_closed.items() if d <= reach_m and d < d_open.get(n, np.inf)}
    params = dict(method="routes", per_zone_stratum=per_zone_stratum, separation_m=separation_m, reach_m=reach_m,
                  buffer_m=buffer_m, min_endpoint_m=min_endpoint_m, max_overlap=max_overlap, min_variety=min_variety,
                  order_seed=int(seed), min_gap_m=round(gap, 1), **candidates)
    return Split(_buffer(graph, closed_nodes, edges, buffer_m), _pairs_frame(corridors, nodes, edges.crs), params)


def by_sectors(edges, nodes, measured, *, rideable, sector_m=3000, per_zone_stratum=8, buffer_m=150,
               min_inside=0.9, min_endpoint_m=400, max_overlap=0.25, min_variety=5, order_seeds=range(8), cache=None,
               **candidates):
    """Checkerboard of ``sector_m`` squares; OD whose corridor lies at least ``min_inside`` in one colour."""
    c = edges.geometry.centroid
    closed_edge = ((c.x // sector_m).astype(int) + (c.y // sector_m).astype(int)) % 2 == 1
    cands = candidate_corridors(edges[rideable], nodes, measured[rideable], min_variety=min_variety, cache=cache, **candidates)
    length = edges.length_m
    share = np.array([length.loc[list(e)][closed_edge.loc[list(e)]].sum() / length.loc[list(e)].sum() for e in cands.edges])
    cands = cands.assign(fixed=np.where(share >= min_inside, "closed", np.where(share <= 1 - min_inside, "open", None)))
    cands = cands[cands.fixed.notna()].reset_index(drop=True)
    corridors, gap, seed = _best(cands, nodes, order_seeds, per_zone_stratum=per_zone_stratum, separation_m=0,
                                 min_endpoint_m=min_endpoint_m, max_overlap=max_overlap)
    graph, _ = _simple_graph(edges, directed=False)
    closed_nodes = set(edges.u[closed_edge]) & set(edges.v[closed_edge])
    params = dict(method="sectors", sector_m=sector_m, per_zone_stratum=per_zone_stratum, buffer_m=buffer_m,
                  min_inside=min_inside, min_endpoint_m=min_endpoint_m, max_overlap=max_overlap, min_variety=min_variety,
                  order_seed=int(seed), **candidates)
    return Split(_buffer(graph, closed_nodes, edges, buffer_m), _pairs_frame(corridors, nodes, edges.crs), params)


SPLITS = {"routes": by_routes, "sectors": by_sectors}
