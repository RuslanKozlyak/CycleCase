"""Route maps for the notebook: one route coloured by several edge features side by side."""

from __future__ import annotations

from dataclasses import dataclass, field
import html

from branca.colormap import LinearColormap
from branca.element import Element, Figure, MacroElement
import folium
from folium.elements import JSCSSMixin
from folium.template import Template
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from .config import COMFORT_PRESETS
from .routing import _oriented_coordinates


ROUTE_OUTLINES = {
    "shortest": "#111827",
    "balanced": "#9333ea",
    "fast": "#64748b",
    "min_turns": "#4d7c0f",
    "avenues": "#2563eb",
    "quiet": "#ec4899",
    "bike_infra": "#0d9488",
    "smooth": "#f59e0b",
    "nature": "#7c2d12",
}
ROUTE_LABELS = {
    "shortest": "кратчайший",
    **{key: preset["label"] for key, preset in COMFORT_PRESETS.items()},
}
BASEMAPS = {
    "OSM, серая": {"tiles": "OpenStreetMap", "className": "grey-tiles"},
    "OSM, цветная": {"tiles": "OpenStreetMap"},
    "Спутник (Esri)": {"tiles": "Esri.WorldImagery"},
    "Рельеф (OpenTopoMap)": {"tiles": "OpenTopoMap"},
}
# Heatmap from good (green) to bad (red).
GOOD_BAD_COLORS = ["#16a34a", "#84cc16", "#facc15", "#f97316", "#dc2626"]
ROAD_CLASS_COLORS = {
    "велодорожка": "#16a34a",
    "крупная + велополоса": "#f59e0b",
    "крупная дорога": "#dc2626",
    "улица": "#64748b",
    "тротуар/пешеходная": "#ec4899",
    "тропа/грунтовка": "#92400e",
}
SURFACE_CLASS_COLORS = {
    "асфальт/бетон": "#64748b",
    "плитка": "#22c55e",
    "брусчатка": "#f59e0b",
    "грунт/гравий": "#92400e",
    "другое": "#a8a29e",
    "не указано": "#cbd5e1",
}


@dataclass(frozen=True)
class Panel:
    """What one map of the grid shows: a feature column and how to colour it."""

    title: str
    column: str
    categories: dict[str, str] = field(default_factory=dict)
    log: bool = False
    low_label: str = "Низкая"
    high_label: str = "Высокая"
    # False: high values are bad (red), e.g. discomfort. True: high is good (green), e.g. popularity.
    higher_is_better: bool = False

    @property
    def colors(self) -> list[str]:
        return GOOD_BAD_COLORS[::-1] if self.higher_is_better else GOOD_BAD_COLORS


class _MapControl(MacroElement):
    """Static HTML box placed in a Leaflet map corner."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: {{ this.position|tojson }}});
        {{ this.get_name() }}.onAdd = function () {
            var div = L.DomUtil.create('div');
            div.style.cssText = 'background:rgba(255,255,255,.92);padding:6px 8px;border-radius:6px;'
                + 'font:12px/1.35 system-ui,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.25);';
            div.innerHTML = {{ this.html|tojson }};
            return div;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )

    def __init__(self, html_text: str, position: str):
        super().__init__()
        self.html = html_text
        self.position = position


class _SyncMaps(JSCSSMixin, MacroElement):
    """Keep view, visible routes and basemap identical across all maps of the grid."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var entries = [
            {%- for entry in this.entries %}
                {map: {{ entry.map }},
                 overlays: { {%- for name, layer in entry.overlays.items() %}{{ name|tojson }}: {{ layer }},{% endfor %} },
                 bases: { {%- for name, layer in entry.bases.items() %}{{ name|tojson }}: {{ layer }},{% endfor %} }},
            {%- endfor %}
            ];
            entries.forEach(function (source) {
                entries.forEach(function (target) {
                    if (target !== source) { source.map.sync(target.map); }
                });
                // Programmatic add/remove fires the same events on the target; hasLayer checks stop the echo.
                source.map.on('overlayadd overlayremove', function (event) {
                    entries.forEach(function (target) {
                        var layer = target.overlays[event.name];
                        if (target === source || !layer) { return; }
                        var shown = target.map.hasLayer(layer);
                        if (event.type === 'overlayadd' && !shown) { target.map.addLayer(layer); }
                        if (event.type === 'overlayremove' && shown) { target.map.removeLayer(layer); }
                    });
                });
                source.map.on('baselayerchange', function (event) {
                    entries.forEach(function (target) {
                        var layer = target.bases[event.name];
                        if (target === source || !layer || target.map.hasLayer(layer)) { return; }
                        target.map.addLayer(layer);
                        Object.keys(target.bases).forEach(function (name) {
                            if (target.bases[name] !== layer) { target.map.removeLayer(target.bases[name]); }
                        });
                    });
                });
            });
        })();
        {% endmacro %}
        """
    )
    default_js = [("Leaflet.Sync", "https://cdn.jsdelivr.net/gh/jieter/Leaflet.Sync/L.Map.Sync.min.js")]

    def __init__(self, entries):
        super().__init__()
        self.entries = entries


def _number(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") if abs(value) >= 100 else f"{value:.3g}"


def _gradient_legend(panel: Panel, low: float, high: float) -> str:
    unit = " (лог. шкала)" if panel.log else ""
    return (
        f"<b>{html.escape(panel.title)}</b><br>"
        f"{panel.low_label} "
        f"<span style='display:inline-block;width:90px;height:9px;vertical-align:middle;"
        f"background:linear-gradient(90deg,{','.join(panel.colors)})'></span> "
        f"{panel.high_label}<br><span style='color:#555'>{_number(low)} … {_number(high)}{unit}</span>"
    )


def _category_legend(panel: Panel, present: set[str]) -> str:
    items = "".join(
        f"<div><span style='display:inline-block;width:18px;height:5px;margin-right:5px;"
        f"vertical-align:middle;background:{color}'></span>{html.escape(label)}</div>"
        for label, color in panel.categories.items()
        if label in present
    )
    return f"<b>{html.escape(panel.title)}</b>{items}"


def _route_legend(routes) -> str:
    shortest = routes.loc[routes["algorithm"] == "shortest", "route_length_m"]
    base = float(shortest.iloc[0]) if len(shortest) else np.nan
    rows = []
    for row in routes.itertuples():
        detour = f", +{100 * (row.route_length_m / base - 1):.0f}%" if row.algorithm != "shortest" and base else ""
        rows.append(
            f"<div><span style='display:inline-block;width:18px;height:9px;margin-right:5px;"
            f"vertical-align:middle;border:3px solid {ROUTE_OUTLINES.get(row.algorithm, '#64748b')}'></span>"
            f"{html.escape(ROUTE_LABELS.get(row.algorithm, row.algorithm))}: "
            f"{row.route_length_m / 1000:.1f} км{detour}</div>"
        )
    return "<b>Маршрут (обводка)</b>" + "".join(rows)


def _to_latlon(coordinates, to_wgs84, tolerance_m: float = 3.0):
    line = LineString(coordinates).simplify(tolerance_m, preserve_topology=False)
    lon, lat = to_wgs84.transform(*line.xy)
    return list(zip(lat, lon))


def _colour_runs(graph, edge_route, colours, labels):
    """Merge consecutive edges of the same colour into one projected coordinate run."""

    runs = []
    for edge in map(tuple, edge_route):
        points = _oriented_coordinates(graph, edge)
        if runs and runs[-1][0] == colours[edge]:
            runs[-1][2].extend(points[1:])
        else:
            runs.append((colours[edge], labels[edge], list(points)))
    return runs


def route_feature_grid(
    graph,
    routes,
    features,
    panels: list[Panel],
    start_latlon: tuple[float, float],
    end_latlon: tuple[float, float],
    *,
    height_px: int = 860,
) -> Figure:
    """Grid of synchronised maps; each colours the same routes by a different edge feature.

    ``routes`` needs ``algorithm``, ``edge_route`` and ``route_length_m``; ``features`` is
    indexed by ``(u, v, key)`` like :func:`cycle_routing.comfort.comfort_feature_table`.
    """

    to_wgs84 = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    route_edges = sorted({tuple(edge) for edges in routes["edge_route"] for edge in edges})
    route_features = features.loc[route_edges]
    lons, lats = to_wgs84.transform(
        route_features.geometry.bounds[["minx", "maxx"]].to_numpy().ravel(),
        route_features.geometry.bounds[["miny", "maxy"]].to_numpy().ravel(),
    )
    bounds = [[float(np.min(lats)), float(np.min(lons))], [float(np.max(lats)), float(np.max(lons))]]

    figure = Figure(height=f"{height_px}px")
    # Grey basemap by default so that only the route carries colour, as in the reference picture.
    figure.header.add_child(Element(
        "<style>.grey-tiles{filter:grayscale(1) brightness(1.08) contrast(.85)}</style>"
    ))
    columns = 2
    rows = int(np.ceil(len(panels) / columns))
    sync_entries = []
    # Routes are drawn as a cable instead of side-by-side offsets (offsets loop at sharp turns):
    # one feature-coloured core, and one ring per model that passes along the segment.
    core_weight, ring_px = (5, 2) if len(routes) <= 4 else (4, 1)
    for index, panel in enumerate(panels):
        fmap = folium.Map(
            tiles=None,
            width=f"{100 / columns}%",
            height=f"{100 / rows}%",
            left=f"{100 / columns * (index % columns)}%",
            top=f"{100 / rows * (index // columns)}%",
            position="absolute",
            control_scale=True,
        )
        figure.add_child(fmap)
        bases = {}
        for base_index, (base_name, spec) in enumerate(BASEMAPS.items()):
            options = {key: value for key, value in spec.items() if key != "tiles"}
            bases[base_name] = folium.TileLayer(
                spec["tiles"], name=base_name, show=base_index == 0, **options
            ).add_to(fmap)
        values = route_features[panel.column]
        if panel.categories:
            labels = values.astype(str)
            colours = labels.map(panel.categories).fillna("#6b7280")
            legend = _category_legend(panel, set(labels))
        else:
            numeric = values.astype(float).fillna(0.0)
            scaled = np.log1p(numeric) if panel.log else numeric
            low, high = float(scaled.min()), float(scaled.max())
            colormap = LinearColormap(panel.colors, vmin=low, vmax=high if high > low else low + 1)
            colours = scaled.map(colormap)
            labels = numeric.map(lambda value: f"{value:.3g}")
            legend = _gradient_legend(panel, float(numeric.min()), float(numeric.max()))
        colours, labels = colours.to_dict(), labels.to_dict()

        overlays = {}
        for order, row in enumerate(routes.itertuples()):
            # One pane per route with a fixed z-order keeps the widest ring at the bottom
            # no matter in which order routes are switched on and off.
            pane = f"route-{order}"
            folium.map.CustomPane(pane, z_index=410 + order, pointer_events=True).add_to(fmap)
            name = ROUTE_LABELS.get(row.algorithm, row.algorithm)
            group = folium.FeatureGroup(name=name, overlay=True, show=True).add_to(fmap)
            runs = _colour_runs(graph, row.edge_route, colours, labels)
            full_line = [runs[0][2][0], *(point for run in runs for point in run[2][1:])]
            lines = [folium.PolyLine(
                _to_latlon(full_line, to_wgs84),
                color=ROUTE_OUTLINES.get(row.algorithm, "#64748b"),
                weight=core_weight + 2 * ring_px * (len(routes) - order),
                opacity=1.0,
            )]
            lines += [
                folium.PolyLine(
                    _to_latlon(points, to_wgs84), color=colour, weight=core_weight, opacity=1.0,
                    tooltip=f"{name}: {label}",
                )
                for colour, label, points in runs
            ]
            for line in lines:
                line.options["pane"] = pane  # folium's path_options drops unknown keys such as pane
                line.add_to(group)
            overlays[name] = group

        folium.Marker(start_latlon, tooltip="старт", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
        folium.Marker(end_latlon, tooltip="финиш", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
        fmap.add_child(_MapControl(legend, "bottomright"))
        if index == 0:
            fmap.add_child(_MapControl(_route_legend(routes), "bottomleft"))
        folium.LayerControl(collapsed=True).add_to(fmap)
        fmap.fit_bounds(bounds)
        sync_entries.append({
            "map": fmap.get_name(),
            "overlays": {name: layer.get_name() for name, layer in overlays.items()},
            "bases": {name: layer.get_name() for name, layer in bases.items()},
        })

    figure.add_child(_SyncMaps(sync_entries))
    return figure
