"""
Mumbai Waterlogging Tracker — UI-only single-file version.

No backend/API integration. Sensor readings are mock data generated on
each refresh purely for presentation purposes. Community photo reports are
stored in-memory (st.session_state) for the current session only.

Run with:
    pip install streamlit folium streamlit-folium pandas Pillow
    streamlit run wen.py
"""

import io
import random
import requests
import html
import threading
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(
    page_title="Mumbai Waterlogging Tracker",
    page_icon="\U0001F327",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global styling: slightly larger font, black text everywhere
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px !important;
        color: #000000 !important;
    }
    p, span, label, div, li, td, th, a {
        color: #000000 !important;
    }
    h1 { font-size: 2.3rem !important; color: #000000 !important; }
    h2 { font-size: 1.8rem !important; color: #000000 !important; }
    h3 { font-size: 1.5rem !important; color: #000000 !important; }
    h4, h5, h6 { color: #000000 !important; }
    .stMarkdown, .stText, .stCaption, .stMetric label, .stMetric div {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 17px !important;
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    /* Color the "How bad is it?" severity slider track: green -> orange -> red */
    div[data-baseweb="slider"] > div > div:first-child {
        background: linear-gradient(
            to right,
            #2ecc71 0%, #2ecc71 33%,
            #e67e22 33%, #e67e22 66%,
            #e74c3c 66%, #e74c3c 100%
        ) !important;
        height: 8px !important;
    }
    div[data-baseweb="slider"] > div > div:nth-child(2) {
        background: transparent !important;
    }
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #ffffff !important;
        border: 3px solid #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Mock data: known Mumbai waterlogging hotspots
# ---------------------------------------------------------------------------
MUMBAI_HOTSPOTS = [
    {"id": "SEN001", "name": "Hindmata Junction, Dadar", "lat": 19.0176, "lon": 72.8443},
    {"id": "SEN002", "name": "Milan Subway, Santacruz", "lat": 19.0825, "lon": 72.8420},
    {"id": "SEN003", "name": "Andheri Subway", "lat": 19.1197, "lon": 72.8464},
    {"id": "SEN004", "name": "Sion Circle", "lat": 19.0448, "lon": 72.8622},
    {"id": "SEN005", "name": "King's Circle, Matunga", "lat": 19.0270, "lon": 72.8560},
    {"id": "SEN006", "name": "Gandhi Market, Sion", "lat": 19.0430, "lon": 72.8600},
    {"id": "SEN007", "name": "Kurla West", "lat": 19.0728, "lon": 72.8826},
    {"id": "SEN008", "name": "Wadala", "lat": 19.0170, "lon": 72.8570},
    {"id": "SEN009", "name": "Chembur Naka", "lat": 19.0522, "lon": 72.9005},
    {"id": "SEN010", "name": "Parel", "lat": 19.0100, "lon": 72.8400},
    {"id": "SEN011", "name": "Vakola, Santacruz East", "lat": 19.0870, "lon": 72.8560},
    {"id": "SEN012", "name": "Lower Parel", "lat": 18.9960, "lon": 72.8300},
    {"id": "SEN013", "name": "Byculla", "lat": 18.9750, "lon": 72.8330},
    {"id": "SEN014", "name": "Marine Lines", "lat": 18.9430, "lon": 72.8240},
    {"id": "SEN015", "name": "Malad Subway", "lat": 19.1860, "lon": 72.8490},
]
MUMBAI_CENTER = {"lat": 19.0760, "lon": 72.8777}

SEVERITY_THRESHOLDS = {"safe": 15, "moderate": 30}
SEVERITY_COLORS = {
    "safe": "#2ecc71",
    "moderate": "#e67e22",
    "severe": "#e74c3c",
    "unknown": "#95a5a6",
}


def classify_severity(water_level_cm: float) -> str:
    if water_level_cm is None:
        return "unknown"
    if water_level_cm < SEVERITY_THRESHOLDS["safe"]:
        return "safe"
    elif water_level_cm < SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "severe"


def generate_mock_readings():
    now = datetime.now(IST)
    readings = []
    for spot in MUMBAI_HOTSPOTS:
        base = random.choices(
            population=[random.uniform(0, 14), random.uniform(15, 29), random.uniform(30, 55)],
            weights=[50, 30, 20],
            k=1,
        )[0]
        readings.append({
            "sensor_id": spot["id"],
            "name": spot["name"],
            "lat": spot["lat"],
            "lon": spot["lon"],
            "water_level_cm": round(base, 1),
            "battery_pct": random.randint(55, 100),
            "timestamp": now.isoformat(),
        })
    return readings


# ---------------------------------------------------------------------------
# Route-finding: builds a simple road-network graph over the hotspot nodes
# and finds the shortest path from origin to destination that AVOIDS any
# node currently classified as "severe". "moderate" nodes are allowed but
# penalized (weighted higher) so the route prefers "safe" roads when
# possible.
# ---------------------------------------------------------------------------
import math
import heapq

ROUTE_SEVERITY_PENALTY = {"safe": 1.0, "moderate": 1.8, "severe": None, "unknown": 1.2}
K_NEAREST_NEIGHBORS = 4


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_route_graph(readings_by_name, origin_name, dest_name):
    """
    Builds a k-nearest-neighbor graph over all nodes that are safe/moderate,
    plus the origin and destination themselves (even if severe, so they can
    still act as endpoints - they just can't be used as a pass-through for
    other routes). Returns adjacency dict: {name: [(neighbor_name, weight), ...]}
    """
    usable_nodes = []
    for spot in MUMBAI_HOTSPOTS:
        sev = readings_by_name.get(spot["name"], {}).get("severity", "unknown")
        if sev != "severe" or spot["name"] in (origin_name, dest_name):
            usable_nodes.append(spot)

    graph = {n["name"]: [] for n in usable_nodes}

    for a in usable_nodes:
        dists = []
        for b in usable_nodes:
            if a["id"] == b["id"]:
                continue
            d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            dists.append((d, b))
        dists.sort(key=lambda x: x[0])
        for d, b in dists[:K_NEAREST_NEIGHBORS]:
            sev_b = readings_by_name.get(b["name"], {}).get("severity", "unknown")
            penalty = ROUTE_SEVERITY_PENALTY.get(sev_b, 1.2)
            if penalty is None:  # severe and not an endpoint -> already excluded above, skip just in case
                continue
            weight = d * penalty
            graph[a["name"]].append((b["name"], weight, d))
            graph.setdefault(b["name"], [])
            # ensure edge is bidirectional
            if not any(n == a["name"] for n, _, _ in graph[b["name"]]):
                sev_a = readings_by_name.get(a["name"], {}).get("severity", "unknown")
                penalty_a = ROUTE_SEVERITY_PENALTY.get(sev_a, 1.2) or 1.2
                graph[b["name"]].append((a["name"], d * penalty_a, d))

    return graph


def find_safe_route(readings_by_name, origin_name, dest_name):
    """
    Dijkstra shortest path avoiding severe nodes. Returns
    (path_names, total_distance_km) or (None, None) if no route exists.
    """
    graph = build_route_graph(readings_by_name, origin_name, dest_name)

    if origin_name not in graph or dest_name not in graph:
        return None, None

    dist = {n: float("inf") for n in graph}
    dist[origin_name] = 0
    real_dist = {n: float("inf") for n in graph}
    real_dist[origin_name] = 0
    prev = {n: None for n in graph}
    visited = set()
    pq = [(0, origin_name)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dest_name:
            break
        for v, w, real_w in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                real_dist[v] = real_dist[u] + real_w
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist.get(dest_name, float("inf")) == float("inf"):
        return None, None

    path = []
    node = dest_name
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, round(real_dist[dest_name], 2)


def get_road_geometry(waypoint_latlon):
    """
    Calls the free OSRM public routing API to snap a sequence of waypoints
    (list of (lat, lon) tuples, in order) onto real roads and returns the
    actual road-following geometry.

    Returns (road_coords, road_distance_km, road_duration_min, source) where
    road_coords is a list of (lat, lon) points tracing the real roads, and
    source is "road" if OSRM succeeded, or "straight_line" if it fell back
    to connecting the waypoints directly (e.g. no internet, or OSRM
    unreachable/rate-limited).
    """
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in waypoint_latlon)
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"
    params = {"overview": "full", "geometries": "geojson"}

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            geom = route["geometry"]["coordinates"]  # list of [lon, lat]
            road_coords = [(lat, lon) for lon, lat in geom]
            distance_km = round(route["distance"] / 1000, 2)
            duration_min = round(route["duration"] / 60, 1)
            return road_coords, distance_km, duration_min, "road"
    except Exception:
        pass

    # Fallback: straight lines between the original waypoints
    return list(waypoint_latlon), None, None, "straight_line"


# ---------------------------------------------------------------------------
# Shared app-wide store (sensor data + community reports).
#
# Uses st.cache_resource so ALL devices/browsers connected to this running
# server share the SAME data - this is what fixes the "different readings
# on different devices" and "photo not showing up for others" issues.
# Everything here is in-memory only: it resets if the server restarts, and
# is only shared across users hitting this exact running instance (not
# across separate deployments) - a real production version would use a
# database instead.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_app_store():
    return {
        "sensor_data": generate_mock_readings(),
        "sensor_last_refresh": datetime.now(IST),
        "community_reports": [],
        "lock": threading.Lock(),
    }


app_store = get_app_store()

# ---------------------------------------------------------------------------
# Shared live chat store.
#
# st.cache_resource creates ONE object shared by every user/session
# connected to this running server process (unlike st.session_state, which
# is private per browser tab). This is what makes the chat genuinely
# multi-user "live" - as long as everyone is using the same running
# `streamlit run` instance.
#
# Limitations (be upfront about these):
#   - Resets if the server process restarts (in-memory only, no database).
#   - Only shared across users hitting this exact server instance - if you
#     deploy multiple instances behind a load balancer, they won't share
#     chat history unless you swap this for a real database.
#   - Not push-based real-time: a user sees new messages when they send a
#     message themselves or click "Refresh messages" (no websockets here).
# ---------------------------------------------------------------------------
MAX_MESSAGES_PER_ROOM = 200
MAX_REPORTS = 100  # cap stored community reports/photos to limit shared server memory use


@st.cache_resource
def get_chat_store():
    return {"messages": {}, "lock": threading.Lock()}


CHAT_GENERAL_KEY = "general"
CHAT_GENERAL_LABEL = "\U0001F306 General \u2014 All Mumbai"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F327 Waterlogging Tracker")
    st.caption("Mumbai \u2014 UI preview (mock data)")

    if st.button("\U0001F504 Refresh sensor data", use_container_width=True):
        with app_store["lock"]:
            app_store["sensor_data"] = generate_mock_readings()
            app_store["sensor_last_refresh"] = datetime.now(IST)
        st.rerun()

    st.info("This is a **UI-only preview** \u2014 all sensor readings shown are mock data for demonstration.", icon="\u2139\uFE0F")

    st.caption(f"Last refresh: {app_store['sensor_last_refresh'].strftime('%d %b %Y, %H:%M:%S')} IST")

    st.divider()
    area_names = ["All areas"] + [h["name"] for h in MUMBAI_HOTSPOTS]
    selected_area = st.selectbox("Filter by area", area_names)

    st.divider()
    st.markdown(
        """
        <div style="font-size:19px; line-height:1.9; color:#000000;">
        <b>Severity thresholds (water depth):</b><br>
        🟢 Safe &lt; 15cm<br>
        🟠 Moderate 15-30cm<br>
        🔴 Severe &gt; 30cm
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
df = pd.DataFrame(app_store["sensor_data"])
df["severity"] = df["water_level_cm"].apply(classify_severity)
df["color"] = df["severity"].map(SEVERITY_COLORS)

if selected_area != "All areas":
    df_view = df[df["name"] == selected_area]
    reports_view = [r for r in app_store["community_reports"] if r["location_name"] == selected_area]
else:
    df_view = df
    reports_view = app_store["community_reports"]

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
st.title("Mumbai Waterlogging Detection")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sensors reporting", len(df))
m2.metric("Severe zones", int((df["severity"] == "severe").sum()))
m3.metric("Moderate zones", int((df["severity"] == "moderate").sum()))
m4.metric("Community reports", len(app_store["community_reports"]))

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_map, tab_sensors, tab_report, tab_nav, tab_chat, tab_about = st.tabs(
    ["\U0001F5FA\uFE0F Live Map", "\U0001F4CA Sensor Data", "\U0001F4F8 Report Waterlogging",
     "\U0001F9ED Navigation", "\U0001F4AC Live Chat", "\u2139\uFE0F About"]
)

# --- Live Map ---------------------------------------------------------------
with tab_map:
    st.subheader("Live situation map")
    st.caption("🔵 markers are sensor readings. 📷 markers are community photo reports.")

    fmap = folium.Map(location=[MUMBAI_CENTER["lat"], MUMBAI_CENTER["lon"]], zoom_start=11, tiles="CartoDB positron")

    for _, row in df_view.iterrows():
        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"Water level: {row['water_level_cm']} cm<br>"
            f"Severity: {row['severity'].capitalize()}<br>"
            f"Battery: {row.get('battery_pct', 'N/A')}%"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row["name"],
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.85,
            weight=2,
        ).add_to(fmap)

    for rep in reports_view:
        popup_html = (
            f"<b>Community report</b><br>{rep['location_name']}<br>"
            f"Severity (self-reported): {rep['severity'].capitalize()}<br>"
            f"{rep['timestamp'].strftime('%d %b, %H:%M')}<br>"
            + (f"\"{rep['caption']}\"" if rep.get("caption") else "")
        )
        folium.Marker(
            location=[rep["lat"], rep["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Photo report: {rep['location_name']}",
            icon=folium.Icon(color="blue", icon="camera", prefix="fa"),
        ).add_to(fmap)

    st_folium(fmap, width=None, height=520, returned_objects=[])

# --- Sensor Data --------------------------------------------------------
with tab_sensors:
    st.subheader("Sensor readings")
    display_df = df_view[["sensor_id", "name", "water_level_cm", "severity", "battery_pct", "timestamp"]].rename(
        columns={
            "sensor_id": "Sensor ID",
            "name": "Location",
            "water_level_cm": "Water level (cm)",
            "severity": "Severity",
            "battery_pct": "Battery (%)",
            "timestamp": "Timestamp",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Water level by location")
    import altair as alt

    severity_order = ["safe", "moderate", "severe"]
    color_range = [SEVERITY_COLORS[s] for s in severity_order]

    bar_chart = (
        alt.Chart(df_view)
        .mark_bar()
        .encode(
            x=alt.X("name:N", title="Location", sort="-y"),
            y=alt.Y("water_level_cm:Q", title="Water level (cm)"),
            color=alt.Color(
                "severity:N",
                title="Severity",
                scale=alt.Scale(domain=severity_order, range=color_range),
            ),
            tooltip=["name", "water_level_cm", "severity"],
        )
        .properties(height=400)
    )
    st.altair_chart(bar_chart, use_container_width=True)

# --- Community Reports / Photo Upload -----------------------------------
with tab_report:
    form_col, gallery_col = st.columns([1, 1])

    with form_col:
        st.subheader("Report waterlogging in your area")
        st.caption("Photos and reports are stored for this session only (UI demo — no backend).")

        with st.form("report_form", clear_on_submit=True):
            loc_choice = st.selectbox(
                "Location",
                [h["name"] for h in MUMBAI_HOTSPOTS],
                key="report_loc",
            )
            severity_choice = st.select_slider(
                "How bad is it?",
                options=["safe", "moderate", "severe"],
                value="moderate",
                format_func=lambda s: s.capitalize(),
            )
            reporter_name = st.text_input("Your name (optional)", placeholder="Anonymous")
            caption = st.text_area("Description", placeholder="e.g. Knee-deep water outside the station entrance")

            photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
            if photo is not None:
                st.image(photo, caption="Preview — this is what will be submitted", use_column_width=True)

            submitted = st.form_submit_button("Submit report", use_container_width=True)

            if submitted:
                loc = next(h for h in MUMBAI_HOTSPOTS if h["name"] == loc_choice)
                image_bytes = photo.read() if photo is not None else None
                with app_store["lock"]:
                    app_store["community_reports"].insert(0, {
                        "location_name": loc["name"],
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "severity": severity_choice,
                        "caption": caption,
                        "reporter_name": reporter_name or "Anonymous",
                        "image_bytes": image_bytes,
                        "timestamp": datetime.now(IST),
                    })
                    if len(app_store["community_reports"]) > MAX_REPORTS:
                        del app_store["community_reports"][MAX_REPORTS:]
                st.success("Thanks! Your report has been added to the map and gallery for everyone.")
                st.rerun()

    with gallery_col:
        st.subheader("Photos reported by others")
        st.caption(f"{len(app_store['community_reports'])} report(s) \u2014 visible to everyone using this app.")

        if st.button("\U0001F504 Refresh reports", key="reports_refresh_btn"):
            st.rerun()

        if not app_store["community_reports"]:
            st.info("No community reports yet. Be the first to submit one on the left.")
        else:
            for rep in app_store["community_reports"]:
                with st.container(border=True):
                    if rep["image_bytes"]:
                        img = Image.open(io.BytesIO(rep["image_bytes"]))
                        st.image(img, use_column_width=True)
                    else:
                        st.caption("No photo attached")
                    st.markdown(f"**{rep['location_name']}** \u2014 {rep['severity'].capitalize()}")
                    st.caption(f"By {rep['reporter_name']} \u00b7 {rep['timestamp'].strftime('%d %b %Y, %H:%M')} IST")
                    if rep["caption"]:
                        st.write(rep["caption"])

# --- Navigation -------------------------------------------------------------
with tab_nav:
    st.subheader("Get a route that avoids severe waterlogging")
    st.caption(
        "Pick your origin and destination. The route will avoid any point currently "
        "marked 🔴 Severe, prefer 🟢 Safe roads, and will only use 🟠 Moderate roads "
        "if needed. (Demo routes are calculated between known hotspot points, not "
        "an actual street-level road network.)"
    )

    readings_by_name = {row["name"]: row for _, row in df.iterrows()}
    hotspot_names = [h["name"] for h in MUMBAI_HOTSPOTS]

    nc1, nc2, nc3 = st.columns([2, 2, 1])
    with nc1:
        origin_name = st.selectbox("Origin", hotspot_names, index=0, key="nav_origin")
    with nc2:
        default_dest_index = 1 if len(hotspot_names) > 1 else 0
        dest_name = st.selectbox("Destination", hotspot_names, index=default_dest_index, key="nav_dest")
    with nc3:
        st.write("")
        st.write("")
        find_route_clicked = st.button("Find route", use_container_width=True)

    if find_route_clicked:
        if origin_name == dest_name:
            st.warning("Origin and destination are the same location.")
        else:
            path, total_km = find_safe_route(readings_by_name, origin_name, dest_name)
            st.session_state["nav_result"] = {
                "path": path, "total_km": total_km,
                "origin": origin_name, "dest": dest_name,
            }

    nav_result = st.session_state.get("nav_result")
    if nav_result and nav_result["origin"] == origin_name and nav_result["dest"] == dest_name:
        path = nav_result["path"]
        total_km = nav_result["total_km"]

        if path is None:
            st.error(
                f"No route avoiding severe waterlogging could be found between "
                f"**{origin_name}** and **{dest_name}** right now. Consider waiting "
                f"for water levels to drop, or using an alternate mode of transport."
            )
        else:
            origin_sev = readings_by_name[origin_name]["severity"]
            dest_sev = readings_by_name[dest_name]["severity"]
            if origin_sev == "severe" or dest_sev == "severe":
                st.warning(
                    "Your origin or destination itself currently has **severe** "
                    "waterlogging. The route below gets as close as possible via "
                    "safe/moderate roads, but exercise caution at the final stretch."
                )

            waypoint_coords = [
                (readings_by_name[stop]["lat"], readings_by_name[stop]["lon"]) for stop in path
            ]
            with st.spinner("Fetching real road directions..."):
                road_coords, road_km, road_min, geom_source = get_road_geometry(waypoint_coords)

            if geom_source == "road":
                st.success(
                    f"Route found \u2014 **{road_km} km** by road, approx. **{road_min} min** "
                    f"driving, across {len(path)} waypoint(s)."
                )
            else:
                st.success(f"Route found \u2014 approx. **{total_km} km** (straight-line estimate) across {len(path)} waypoint(s).")
                st.caption(
                    "⚠️ Could not reach the road-routing service, so this route is shown as "
                    "straight lines between waypoints rather than actual roads. Check your "
                    "internet connection and try again."
                )

            steps_md = ""
            for i, stop in enumerate(path):
                sev = readings_by_name[stop]["severity"]
                icon = {"safe": "🟢", "moderate": "🟠", "severe": "🔴"}.get(sev, "⚪")
                steps_md += f"{i + 1}. {icon} **{stop}** ({sev.capitalize()})\n"
            st.markdown(steps_md)

            route_map = folium.Map(
                location=[MUMBAI_CENTER["lat"], MUMBAI_CENTER["lon"]], zoom_start=11, tiles="CartoDB positron"
            )

            # all sensor points, greyed if severe/blocked
            for _, row in df.iterrows():
                is_on_path = row["name"] in path
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=9 if is_on_path else 6,
                    tooltip=f"{row['name']} ({row['severity'].capitalize()})",
                    color=row["color"],
                    fill=True,
                    fill_color=row["color"],
                    fill_opacity=0.9 if is_on_path else 0.4,
                    weight=3 if is_on_path else 1,
                ).add_to(route_map)

            coords = road_coords
            folium.PolyLine(
                coords, color="#2b6cb0", weight=5, opacity=0.85,
                dash_array=None if geom_source == "road" else "8,6",
            ).add_to(route_map)

            folium.Marker(
                waypoint_coords[0], tooltip=f"Start: {origin_name}",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(route_map)
            folium.Marker(
                waypoint_coords[-1], tooltip=f"End: {dest_name}",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
            ).add_to(route_map)

            st_folium(route_map, width=None, height=480, returned_objects=[])
    else:
        st.info("Choose an origin and destination, then click **Find route**.")

# --- Live Chat ---------------------------------------------------------------
with tab_chat:
    st.subheader("Live community chat")
    st.caption(
        "Chat with others using this app right now \u2014 pick the general Mumbai-wide "
        "room, or an area-specific room. New messages appear when you send one or "
        "click **Refresh messages** (this demo doesn't push updates automatically)."
    )

    chat_store = get_chat_store()
    messages_by_room = chat_store["messages"]
    chat_lock = chat_store["lock"]

    room_options = [CHAT_GENERAL_LABEL] + [h["name"] for h in MUMBAI_HOTSPOTS]
    selected_room_label = st.selectbox("Chat room", room_options, key="chat_room_select")
    room_key = CHAT_GENERAL_KEY if selected_room_label == CHAT_GENERAL_LABEL else selected_room_label

    with chat_lock:
        messages_by_room.setdefault(room_key, [])
        room_messages = list(messages_by_room[room_key])

    top_c1, top_c2 = st.columns([1, 3])
    with top_c1:
        if st.button("\U0001F504 Refresh messages", key="chat_refresh_btn", use_container_width=True):
            st.rerun()
    with top_c2:
        st.caption(f"{len(room_messages)} message(s) in **{selected_room_label}**.")

    # --- message display ---
    if not room_messages:
        st.info("No messages yet in this room. Be the first to say something!")
    else:
        rows_html = []
        for msg in room_messages:
            safe_name = html.escape(msg["name"])
            safe_text = html.escape(msg["text"])
            time_str = msg["timestamp"].strftime("%d %b, %H:%M")
            rows_html.append(
                f"<div style='margin:6px 0; padding:8px 12px; background:#f5f5f5; "
                f"border-radius:8px;'>"
                f"<span style='font-weight:600; color:#000000;'>{safe_name}</span> "
                f"<span style='font-size:12px; color:#555555;'>{time_str}</span><br>"
                f"<span style='color:#000000;'>{safe_text}</span>"
                f"</div>"
            )
        chat_html = (
            "<div style='max-height:360px; overflow-y:auto; padding:10px; "
            "border:1px solid #dddddd; border-radius:10px; background:#ffffff;'>"
            + "".join(rows_html)
            + "</div>"
        )
        st.markdown(chat_html, unsafe_allow_html=True)

    # --- message input ---
    with st.form("chat_send_form", clear_on_submit=True):
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            chat_name = st.text_input(
                "Your name", value=st.session_state.get("chat_name", ""),
                placeholder="Anonymous", key="chat_name_input",
            )
        with fc2:
            chat_text = st.text_input(
                "Message", placeholder="Type your message and hit Send...",
                key="chat_text_input",
            )
        send_clicked = st.form_submit_button("Send", use_container_width=True)

        if send_clicked:
            if chat_text.strip():
                with chat_lock:
                    room_list = messages_by_room.setdefault(room_key, [])
                    room_list.append({
                        "name": chat_name.strip() or "Anonymous",
                        "text": chat_text.strip(),
                        "timestamp": datetime.now(IST),
                    })
                    if len(room_list) > MAX_MESSAGES_PER_ROOM:
                        del room_list[: len(room_list) - MAX_MESSAGES_PER_ROOM]
                st.session_state["chat_name"] = chat_name
                st.rerun()
            else:
                st.warning("Type a message before sending.")

# --- About ----------------------------------------------------------------
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
This is a **UI-only preview** of a Mumbai waterlogging tracker, combining:

1. **Sensor readings** (currently mock/demo data, shown on the map & table)
2. **Community photo reports** — anyone can upload a photo and mark severity
   at a location, shown alongside sensor data on the map.
3. **Navigation** — pick an origin and destination and get a route that
   avoids severely waterlogged points, preferring safe roads and using
   moderate ones only when necessary.
4. **Live Chat** — a general Mumbai-wide room plus one room per area, shared
   by everyone currently using this running app instance.

**This build has no backend:**
- Sensor readings are randomly generated each time you hit "Refresh sensor
  data" — purely to make the UI look realistic. There is no real sensor
  connection.
- Community reports/photos are stored only in the browser session
  (`st.session_state`) and reset when the app restarts.
- Photos are **not** analyzed automatically — severity is self-reported.
- Chat messages are stored in server memory (shared across everyone using
  this running instance) rather than a real database, so they're lost if
  the server restarts, and won't sync across multiple separate deployments.
  Chat also isn't push-based — send a message or click refresh to see new
  ones from others.

To turn this into a real product, the next step is wiring a real sensor
API and a persistent database — ask if you'd like that added back in.
        """
    )
