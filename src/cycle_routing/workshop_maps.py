"""Self-contained Leaflet route explorer: one selected OD and solution, no widget extension needed."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .graph import edge_costs
from .routing import orient_coordinates
from .workshop import features


ASSETS=Path(__file__).with_name("assets")


def plot_od_bank(lab):
    """A compact overview of the fixed topology and all shortest reference routes."""
    import matplotlib.pyplot as plt
    from .workshop import run_routes

    routes = run_routes(lab, {})
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    lab.edges.plot(ax=ax, color="#d1d5db", linewidth=.35)
    routes.plot(ax=ax, column="od_id", cmap="tab20", linewidth=1.5, alpha=.8)
    for row in routes.itertuples():
        center = row.geometry.interpolate(.5, normalized=True)
        ax.annotate(row.od_id, (center.x, center.y), fontsize=7, color="#1f2937")
    ax.set(title="20 фиксированных OD и их кратчайшие маршруты", aspect="equal")
    ax.set_axis_off()
    return fig


def _number(value):
    return float(value) if value is not None and pd.notna(value) else None


def route_explorer(lab, result, *, od_id=None, filename=None):
    """Embed local JS/CSS and road context. Internet is needed only for the optional OSM tiles."""
    from IPython.display import IFrame

    roads=lab.edges.set_index(["u","v","key"],drop=False)
    xy=lab.nodes.set_index("node")[["x","y"]]
    transform=Transformer.from_crs(lab.edges.crs,"EPSG:4326",always_xy=True)
    keys={tuple(int(x) for x in key) for path in result.routes.edge_route for key in path}
    public=pd.Series(np.nan,index=lab.edges.index)
    for values in lab.labels.values(): public=public.combine_first(values)
    counts=dict(zip(zip(lab.edges.u,lab.edges.v,lab.edges.key),public))
    f=features(lab.edges).set_axis(pd.MultiIndex.from_frame(lab.edges[["u","v","key"]]))
    segments={}
    for key in sorted(keys):
        row=roads.loc[key]
        coords=orient_coordinates(row.geometry,tuple(xy.loc[key[0]]))
        lon,lat=transform.transform(*zip(*coords))
        high=str(row.highway) if pd.notna(row.highway) else "нет данных"
        segments[":".join(map(str,key))]={
            "line":[[round(a,6),round(b,6)] for a,b in zip(lat,lon)],
            "highway":high,"surface":str(row.surface) if pd.notna(row.surface) else "нет данных",
            "length":float(row.length_m),"trips":_number(counts[key]),
            "bike":bool(f.loc[key,"bike_infra"]),"major":bool(f.loc[key,"major_road"]),
        }
    models=list(result.routes.algorithm.unique())
    costs={}
    for name in models:
        cost="length_m" if name=="shortest" else result.solutions[name]["cost"]
        values=edge_costs(lab.edges,cost)/lab.edges.length_m
        mapping=dict(zip(zip(lab.edges.u,lab.edges.v,lab.edges.key),values))
        costs[name]={":".join(map(str,key)):round(float(mapping[key]),4) for key in keys}
    routes={}
    for row in result.routes.itertuples():
        routes.setdefault(row.od_id,{})[row.algorithm]={
            "edges":[":".join(map(str,map(int,key))) for key in row.edge_route],
            "length":float(row.route_length_m),"turns":int(row.turns),
        }
    detail={}
    for row in result.per_od.itertuples():
        detail.setdefault(row.od_id,{})[row.algorithm]={
            "coverage":_number(row.coverage_pct),"gain":_number(row.gain),"points":_number(row.points),
            "included":bool(row.in_evaluation),"reason":row.reason,
        }
    context=lab.edges.copy()
    if len(context) > 100_000:
        # An external-city graph can be much larger than the route corridor.
        bounds=result.routes.total_bounds
        context=context.cx[bounds[0]-1500:bounds[2]+1500, bounds[1]-1500:bounds[3]+1500]
    context["pair"]=[tuple(sorted((u,v))) for u,v in zip(context.u,context.v)]
    context=context.drop_duplicates("pair")
    context.geometry=context.geometry.simplify(4,preserve_topology=False)
    context=context.to_crs("EPSG:4326")
    lines=[[[round(lat,6),round(lon,6)] for lon,lat,*_ in geometry.coords] for geometry in context.geometry]
    payload={"segments":segments,"routes":routes,"costs":costs,"detail":detail,"context":lines,
             "models":models,"labels":{name:("кратчайший" if name=="shortest" else result.solutions[name].get("label",name)) for name in models},
             "ods":[od for od in lab.pairs.od_id if od in routes],"first":od_id or next(iter(routes)),
             "split":result.split,"maxTrips":float(public.max()) if public.notna().any() else 1}
    template=(ASSETS/"route_explorer.html").read_text(encoding="utf-8")
    document=(template.replace("__LEAFLET_CSS__",(ASSETS/"leaflet/leaflet.css").read_text(encoding="utf-8"))
              .replace("__LEAFLET_JS__",(ASSETS/"leaflet/leaflet.js").read_text(encoding="utf-8"))
              .replace("__DATA__",json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c")))
    if filename:
        path=Path(filename); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(document,encoding="utf-8")
    return IFrame("about:blank", width="100%", height=660, extras=[
        'title="Сравнение велосипедных маршрутов"',
        f'srcdoc="{html.escape(document,quote=True)}"',
        'style="border:1px solid #cbd5e1;border-radius:8px"', 'loading="lazy"',
    ])
