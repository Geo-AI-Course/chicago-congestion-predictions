"""Chicago Road Congestion — public Streamlit app.

Reads the slim dataset baked out of PostGIS by `src/export.py` (webapp/assets/*)
and renders the interactive congestion map. No database, model, or GDAL needed at
runtime — so it deploys cleanly to Streamlit Community Cloud.

Run locally:
    streamlit run webapp/app.py
"""
import json
from pathlib import Path

import folium
import streamlit as st
from folium.elements import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

ASSETS = Path(__file__).parent / "assets"
CHICAGO = [41.8781, -87.6298]


@st.cache_data
def load(name):
    return json.loads((ASSETS / name).read_text())


st.set_page_config(page_title="Chicago Road Congestion", page_icon="🚦", layout="wide")

segments = load("segments.geojson")
bottlenecks = load("bottlenecks.geojson")
metrics = load("metrics.json")

st.title("🚦 Chicago Road Congestion — Predicted Bottlenecks")
st.caption(
    "Predicted volume/capacity (V/C) congestion scores for Chicago's major road "
    "network, with structural bottlenecks pinpointed. Built from OpenStreetMap + "
    "Chicago traffic counts, spatial features engineered in PostGIS, scored by a "
    "Random Forest regressor."
)

# --- headline metrics -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Road segments", f"{metrics['total_segments']:,}")
c2.metric("Labeled (with counts)", f"{metrics['labeled_segments']:,}",
          f"{metrics['labeled_pct']}%")
c3.metric("Traffic-count records", f"{metrics['traffic_count_records']:,}")
c4.metric("Model R² (held-out)", f"{metrics['model']['r2']:.3f}")

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Problem.** Where do Chicago roads choke relative to their design "
        "capacity, and *why*?\n\n"
        "**Spatial / GeoAI.** OSM road network + traffic signals + city traffic "
        "counts are joined in **PostGIS** (150 m `ST_DWithin` snap, GIST index, "
        "UTM distance calcs), enriched with network betweenness centrality, then "
        "a model predicts a V/C score across **every** segment.\n\n"
        "**Stage reached:** descriptive + **predictive**."
    )
    st.subheader("Model (held-out test)")
    st.write(
        f"- RMSE: `{metrics['model']['rmse']:.4f}`\n"
        f"- MAE: `{metrics['model']['mae']:.4f}`\n"
        f"- R²: `{metrics['model']['r2']:.4f}`"
    )
    st.subheader("Data sources")
    st.markdown(
        "- [OpenStreetMap via osmnx](https://www.openstreetmap.org)\n"
        "- [Chicago Data Portal — traffic counts]"
        "(https://data.cityofchicago.org/resource/pfsx-4n4m.geojson)"
    )
    st.subheader("Legend")
    st.markdown("**Road lines** — green (low V/C) → red (high V/C)")
    st.markdown("**Dots** — top-5% congestion score + a structural cause:")
    st.markdown(
        "<div style='line-height:2'>"
        "<span style='color:#e74c3c;font-size:18px'>●</span> Lane drop ahead<br>"
        "<span style='color:#e67e22;font-size:18px'>●</span> 4+ roads merging<br>"
        "<span style='color:#2980b9;font-size:18px'>●</span> On/off ramp weave<br>"
        "<span style='color:#8e44ad;font-size:18px'>●</span> Traffic signals nearby<br>"
        "<span style='color:#c0392b;font-size:18px'>●</span> High network centrality<br>"
        "</div>",
        unsafe_allow_html=True,
    )

_TOOLTIP_STYLE = (
    "font-size: 14px; "
    "line-height: 1.6; "
    "padding: 8px 10px; "
    "max-width: 280px; "
    "background: rgba(255,255,255,0.95); "
    "border: 1px solid #ccc; "
    "border-radius: 6px; "
    "box-shadow: 2px 2px 6px rgba(0,0,0,0.15);"
)


class SegmentTooltip(MacroElement):
    """Bind per-feature HTML tooltips to a GeoJson layer using inline styles.

    GeoJsonTooltip's CSS-based styling is beaten by Leaflet's own stylesheet.
    This MacroElement bypasses that by calling layer.bindTooltip() with a
    function that returns fully inline-styled HTML — the same mechanism that
    makes folium.Tooltip work on the bottleneck dot markers.
    """

    def __init__(self, layer_name: str):
        super().__init__()
        self._name = "SegmentTooltip"
        self.layer_name = layer_name
        self._template = Template("""
            {% macro script(this, kwargs) %}
            {{ this.layer_name }}.bindTooltip(
                function(layer) {
                    var p = layer.feature.properties;
                    var name = p.name || "unnamed";
                    var score = p.score !== undefined ? parseFloat(p.score).toFixed(3) : "N/A";
                    return (
                        "<div style='font-size:14px;line-height:1.7;"
                        + "min-width:140px;padding:2px 4px'>"
                        + "<b>" + name + "</b><br>"
                        + "V/C score: <b>" + score + "</b>"
                        + "</div>"
                    );
                },
                {sticky: true, opacity: 1.0}
            );
            {% endmacro %}
        """)


# --- map --------------------------------------------------------------------
m = folium.Map(location=CHICAGO, zoom_start=11, tiles="CartoDB positron")

# On-map legend for bottleneck dot colors
_LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 36px; left: 36px;
    z-index: 1000;
    background: rgba(255,255,255,0.95);
    border: 1px solid #bbb;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    line-height: 2;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    pointer-events: none;
">
<b style="font-size:13px">Bottleneck cause</b><br>
<span style="color:#e74c3c;font-size:16px">●</span>&nbsp;Lane drop ahead<br>
<span style="color:#e67e22;font-size:16px">●</span>&nbsp;4+ roads merging<br>
<span style="color:#2980b9;font-size:16px">●</span>&nbsp;On/off ramp weave<br>
<span style="color:#8e44ad;font-size:16px">●</span>&nbsp;Traffic signals nearby<br>
<span style="color:#c0392b;font-size:16px">●</span>&nbsp;High network centrality<br>
<hr style="margin:6px 0;border-color:#ddd">
<span style="color:#27ae60;font-size:16px">━</span>&nbsp;Low congestion (V/C)<br>
<span style="color:#e74c3c;font-size:16px">━</span>&nbsp;High congestion (V/C)<br>
</div>
"""
m.get_root().html.add_child(folium.Element(_LEGEND_HTML))

gj = folium.GeoJson(
    segments,
    name="Predicted congestion",
    style_function=lambda f: {
        "color": f["properties"]["color"],
        "weight": f["properties"]["weight"],
        "opacity": 0.8,
    },
)
gj.add_to(m)
SegmentTooltip(gj.get_name()).add_to(m)

for feat in bottlenecks["features"]:
    lon, lat = feat["geometry"]["coordinates"]
    p = feat["properties"]
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=p["color"],
        weight=1,
        fill=True,
        fill_color=p["color"],
        fill_opacity=0.85,
        tooltip=folium.Tooltip(p["tooltip"], sticky=True, style=_TOOLTIP_STYLE),
    ).add_to(m)

st_folium(m, use_container_width=True, height=680, returned_objects=[])

with st.expander("Static overview map"):
    png = ASSETS / "congestion_static.png"
    if png.exists():
        st.image(str(png), use_column_width=True)
    else:
        st.info("Static map not exported yet — run `python src/export.py`.")
