
from pathlib import Path
from html import escape as html_escape
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, URIRef, Literal
from pyvis.network import Network

from graphrag_ollama import (
    retrieve_sensors,
    retrieve_maintenance,
    detect_anomalies,
    build_hybrid_context,
    answer_with_ollama,
    TEMP_THRESHOLD,
    VIBRATION_THRESHOLD,
)

from document_retrieval import retrieve_documents


st.set_page_config(
    page_title="Industrial Intelligence | GraphRAG",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------
# Styling
# -----------------------------------

st.markdown("""
<style>
.block-container {
    max-width: 1550px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #091B30;
    border-right: 1px solid #263750;
}

div[data-testid="stMetric"] {
    background: #152238;
    padding: 18px;
    border: 1px solid #263750;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

.stButton > button[kind="primary"] {
    background: #0891B2;
    color: white;
    border-radius: 10px;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background: #0E7490;
}

div[data-testid="stChatMessage"] {
    background: #152238;
    border: 1px solid #263750;
    border-radius: 12px;
}

h1, h2, h3 {
    letter-spacing: -0.025em;
}

.subtitle {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* Portfolio UI: consistent deep navy, cyan accents and legible cards */
.stApp { background: #071323; color: #F1F5F9; }
.block-container { max-width: 1550px; padding-top: 1.3rem; }
section[data-testid="stSidebar"] { background: #0B1C31; }
div[data-testid="stMetric"] { background: linear-gradient(145deg,#10243D,#0C1C32); border: 1px solid #24496B; box-shadow: 0 6px 20px #00000022; min-height: 120px; }
div[data-testid="stMetric"]:nth-child(2) { border-color: #9F354F; }
div[data-testid="stChatMessage"] { background: #10243D; border-color: #2B5378; padding: 0.9rem 1rem; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: #22D3EE; }
div[data-testid="stDataFrame"] { border: 1px solid #27435F; border-radius: 10px; overflow: hidden; }
div[data-testid="stChatInput"] { border: 1px solid #0891B2; border-radius: 12px; }
.evidence-card { background: #10243D; border: 1px solid #31516E; border-radius: 12px; padding: 16px; min-height: 160px; }
.evidence-card h4 { margin: 0 0 12px; color: #67E8F9; font-size: 1.04rem; }
.evidence-card p { margin: 7px 0; font-size: .94rem; }
.evidence-card strong { color: #E2E8F0; }
.evidence-card.sensor { border-color: #0D9488; }
.evidence-card.graph { border-color: #3B82F6; }
.evidence-card.docs { border-color: #8B5CF6; }

/* Match the showcase: clean neon accents, dark glass cards, and tables */
div[data-testid="stMetric"] { border-radius: 13px; border: 1px solid #254B70; background: linear-gradient(140deg,#112C4D,#0B1C32); }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { border-color: #9E4059; background: linear-gradient(140deg,#3B1D35,#141D35); }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { border-color: #127B63; background: linear-gradient(140deg,#103B39,#102339); }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { border-color: #64748B; }
div[data-testid="stChatMessage"] { background: #11243C; border: 1px solid #264765; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #243C56; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.showcase-table { width: 100%; border-collapse: collapse; font-size: .92rem; background: #0C192C; }
.showcase-table th { text-align: left; background: #17253A; color: #CBD5E1; }
.showcase-table td,.showcase-table th { padding: 12px 13px; border: 1px solid #253750; }
.showcase-table tr:hover td { background: #15253D; }
.status-attention { color: #FB7185; font-weight: 700; }
.status-normal { color: #34D399; font-weight: 700; }
.status-unknown { color: #94A3B8; font-weight: 700; }
.evidence-strip { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 18px; }
.evidence-pill { padding:9px 12px; background:#123C42; border:1px solid #0D9488; border-radius:8px; color:#CFFAFE; font-size:.82rem; }
.evidence-pill.graph { background:#142D54; border-color:#3B82F6; }
.evidence-pill.doc { background:#2A1F4B; border-color:#8B5CF6; }
.graph-legend { display:flex; flex-wrap:wrap; gap:16px; color:#CBD5E1; font-size:.83rem; margin:12px 0; }
.graph-dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; }

/* Reference-matched illustrated icon tiles and gradient KPI panels */
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:14px 0 22px; }
.kpi-card { min-height:118px; padding:20px 16px; display:flex; align-items:center; gap:13px; border-radius:14px; border:1px solid #2C527C; background:linear-gradient(140deg,#102D50,#0A1B31); }
.kpi-card.red { border-color:#BE4567; background:linear-gradient(140deg,#3C2039,#161B31); }
.kpi-card.green { border-color:#168C71; background:linear-gradient(140deg,#103C39,#0C2333); }
.kpi-card.gray { border-color:#52647D; background:linear-gradient(140deg,#25364E,#122138); }
.kpi-icon { flex:0 0 48px; width:48px; height:48px; display:flex; align-items:center; justify-content:center; color:white; border-radius:12px; background:linear-gradient(145deg,#1387F3,#1758B8); box-shadow:0 6px 18px #0003; }
.red .kpi-icon { background:linear-gradient(145deg,#FF587A,#C72555); }
.green .kpi-icon { background:linear-gradient(145deg,#16BD84,#08785B); }
.gray .kpi-icon { background:linear-gradient(145deg,#7086A3,#465A76); }
.kpi-title { font-size:.84rem; color:#E2E8F0; margin-bottom:5px; white-space:nowrap; }
.kpi-number { font-size:2.05rem; font-weight:750; line-height:1.05; color:#FFF; }
.evidence-card { padding:0; overflow:hidden; min-height:235px; }
.evidence-card h4 { display:flex; align-items:center; gap:10px; padding:14px 15px; margin:0; color:#F8FAFC; background:#075A52; font-size:1rem; }
.evidence-card.graph h4 { background:#153F85; }
.evidence-card.docs h4 { background:#4C2D8B; }
.evidence-card h4 .evidence-icon { display:flex; align-items:center; justify-content:center; width:34px; height:34px; background:#0BAE86; border-radius:8px; color:#FFF; }
.evidence-card.graph h4 .evidence-icon { background:#2978E7; }
.evidence-card.docs h4 .evidence-icon { background:#8050E0; }
.evidence-card .evidence-body { padding:12px 15px 17px; }
.evidence-card .evidence-body p { padding:5px 0; margin:0; border-bottom:1px solid #213750; }
.evidence-card .evidence-body p:last-child { border-bottom:none; }
@media(max-width:1000px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media(max-width:550px) { .kpi-grid { grid-template-columns:1fr; } .kpi-title { white-space:normal; } }

/* Compact fleet table matching the showcase screenshot */
.fleet-table-wrap { width:100%; overflow-x:auto; border:1px solid #263B56; border-radius:9px; margin:12px 0 14px; }
.showcase-table { width:100%; border-collapse:collapse; table-layout:auto; font-size:.86rem; background:#081629; }
.showcase-table thead th { padding:11px 12px; background:#132239; color:#D8E2F0; font-size:.83rem; font-weight:650; white-space:nowrap; }
.showcase-table td { padding:10px 12px; border-bottom:1px solid #233751; color:#E2E8F0; line-height:1.3; }
.showcase-table tr:last-child td { border-bottom:0; }
.showcase-table td:first-child { color:#F8FAFC; font-weight:700; }
.showcase-table td:nth-child(2), .showcase-table td:nth-child(3) { font-variant-numeric:tabular-nums; }
.showcase-table td.reading-alert { color:#FF607B; font-weight:750; }
.showcase-table .status-attention { color:#FB5E7D; font-weight:650; }
.showcase-table .status-unknown { color:#98A8BF; font-weight:600; }
.showcase-table .status-normal { color:#34D399; font-weight:650; }
.showcase-table .status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; background:currentColor; vertical-align:middle; }
.showcase-table .no-data { color:#91A1B7; }
@media (max-width:800px) { .showcase-table { min-width:690px; } }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Helpers
# -----------------------------------

# Small inline SVG icons: consistent across macOS, browsers and screenshots.
# All paths are fixed application assets, not user-generated markup.
ICON_PATHS = {
    "machines": '<rect x="3" y="9" width="18" height="12" rx="2"/><path d="M3 9l5-5 5 5 5-5 3 3"/><path d="M8 14v3m5-3v3m5-3v3"/>',
    "alert": '<path d="M10.3 3.9 2.2 18a2 2 0 0 0 1.7 3h16.2a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 4h.01"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    "sensor": '<path d="M12 19v-7m-5 8h10"/><circle cx="12" cy="8" r="3"/><path d="M6 4a9 9 0 0 0 0 8m12-8a9 9 0 0 1 0 8"/>',
    "graph": '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="19" cy="19" r="2"/><path d="m7 11 10-5M7 13l10 5"/>',
    "bot": '<rect x="4" y="7" width="16" height="13" rx="3"/><path d="M12 3v4m-4 6h.01M16 13h.01M9 17h6"/>',
}

def ui_icon(name, size=22):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.9" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        + ICON_PATHS[name] + '</svg>'
    )


MACHINES = [f"M10{i}" for i in range(1, 6)]


@st.cache_data
def get_fleet_data():
    rows = []

    for machine_id in MACHINES:
        sensors = retrieve_sensors(machine_id)
        maintenance = retrieve_maintenance(machine_id)

        temp = sensors.get("temperature")
        vibration = sensors.get("vibration")

        if temp is None or vibration is None:
            status = "Unknown"
        elif temp > TEMP_THRESHOLD or vibration > VIBRATION_THRESHOLD:
            status = "Attention"
        else:
            status = "Normal"

        rows.append({
            "Machine": machine_id,
            "Temperature (°C)": temp,
            "Vibration (mm/s)": vibration,
            "Status": status,
            "Possible faults": ", ".join(
                sorted({f["fault"] for f in maintenance})
            ) if maintenance else "Not documented",
        })

    return pd.DataFrame(rows)


@st.cache_data
def get_graph_html(machine_id):
    graph = Graph()
    graph.parse("industrial_kg_aligned.ttl", format="turtle")

    # Include both URI representations of the selected machine
    sensor_uri = URIRef(
        f"https://example.org/industrial/"
        f"{machine_id.replace('M', 'M-', 1)}"
    )
    maintenance_uri = URIRef(
        f"http://example.org/industrial/{machine_id}"
    )

    # Traverse outgoing relationships up to 3 hops
    frontier = {sensor_uri, maintenance_uri}
    visited = set()
    triples = set()

    for _ in range(3):
        next_frontier = set()

        for subject in frontier:
            if subject in visited:
                continue

            visited.add(subject)

            for s, p, o in graph.triples((subject, None, None)):
                triples.add((s, p, o))

                if isinstance(o, URIRef):
                    next_frontier.add(o)

        frontier = next_frontier

    net = Network(
        height="550px",
        width="100%",
        directed=True,
        bgcolor="#101827",
        font_color="#F8FAFC",
        cdn_resources="in_line",
    )

    # Hierarchical layout separates related nodes into readable levels.
    # Physics is disabled to prevent nodes collapsing on top of each other.
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 210,
          "nodeSpacing": 165,
          "treeSpacing": 220,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
        }
      },
      "nodes": {
        "shape": "dot",
        "size": 22,
        "font": {
          "size": 17,
          "color": "#F8FAFC",
          "strokeWidth": 3,
          "strokeColor": "#101827"
        }
      },
      "edges": {
        "color": {"color": "#64748B", "highlight": "#22D3EE"},
        "width": 2,
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
        "font": {
          "size": 12,
          "color": "#CBD5E1",
          "strokeWidth": 3,
          "strokeColor": "#101827",
          "align": "middle"
        },
        "smooth": {"enabled": true, "type": "cubicBezier", "forceDirection": "horizontal"}
      },
      "physics": {"enabled": false},
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      }
    }
    """)

    def short_name(value):
        return str(value).split("/")[-1].split("#")[-1]

    machine_nodes = {str(sensor_uri), str(maintenance_uri)}

    for s, p, o in sorted(triples, key=lambda t: tuple(map(str, t))):
        source = str(s)

        source_color = (
            "#22D3EE" if source in machine_nodes else "#60A5FA"
        )

        net.add_node(
            source,
            label=short_name(s),
            title=source,
            color=source_color,
            size=32 if source in machine_nodes else 23,
        )

        if isinstance(o, URIRef):
            target = str(o)

            net.add_node(
                target,
                label=short_name(o),
                title=target,
                color=(
                    "#22D3EE"
                    if target in machine_nodes
                    else "#A78BFA"
                ),
                size=32 if target in machine_nodes else 23,
            )

        else:
            target = f"{source}|{p}|{o}"

            net.add_node(
                target,
                label=(str(o)[:31] + "…" if len(str(o)) > 32 else str(o)),
                title=str(o),
                color="#FBBF24",
                size=17,
            )

        net.add_edge(
            source,
            target,
            label=short_name(p),
            title=short_name(p),
        )

    return net.generate_html()


# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("🏭 Industrial Intelligence")
st.sidebar.caption("GraphRAG Operations Center")
st.sidebar.divider()

selected_machine = st.sidebar.selectbox(
    "Selected machine",
    MACHINES,
    index=1,
)

st.sidebar.divider()
st.sidebar.markdown("**Technology stack**")
st.sidebar.caption(
    "Python · RDFLib · SPARQL · "
    "Sentence Transformers · Ollama · Streamlit"
)

st.sidebar.divider()
st.sidebar.info(
    "Synthetic industrial dataset. "
    "Demonstration thresholds only."
)


# -----------------------------------
# Header
# -----------------------------------

st.title("Industrial GraphRAG Copilot")
st.markdown(
    '<p class="subtitle">'
    'Explainable equipment monitoring and maintenance '
    'assistance powered by Knowledge Graphs and local LLMs.'
    '</p>',
    unsafe_allow_html=True,
)

fleet_df = get_fleet_data()

attention_count = int(
    (fleet_df["Status"] == "Attention").sum()
)

normal_count = int(
    (fleet_df["Status"] == "Normal").sum()
)

unknown_count = int(
    (fleet_df["Status"] == "Unknown").sum()
)

# -----------------------------------
# Tabs
# -----------------------------------

tab_fleet, tab_machine, tab_graph, tab_ai, tab_evidence = st.tabs([
    "Fleet Overview",
    "Machine Details",
    "Knowledge Graph",
    "AI Assistant",
    "Evidence Explorer",
])


# ===================================
# Fleet Overview
# ===================================

with tab_fleet:
    st.subheader("Fleet Overview")

    kpis = [
        ("machines", "Total Machines", len(MACHINES), ""),
        ("alert", "Attention Required", attention_count, "red"),
        ("check", "Normal", normal_count, "green"),
        ("file", "No Data", unknown_count, "gray"),
    ]
    kpi_html = '<div class="kpi-grid">'
    for icon_name, label, count, variant in kpis:
        kpi_html += (
            f'<div class="kpi-card {variant}">'
            f'<span class="kpi-icon">{ui_icon(icon_name, 26)}</span>'
            f'<div><div class="kpi-title">{html_escape(label)}</div>'
            f'<div class="kpi-number">{count}</div></div></div>'
        )
    st.markdown(kpi_html + '</div>', unsafe_allow_html=True)

    st.divider()

    # Compact, color-coded table modeled on the portfolio showcase.
    table_rows = []
    for _, machine in fleet_df.iterrows():
        status = str(machine["Status"])
        status_class = {"Attention": "attention", "Normal": "normal"}.get(status, "unknown")
        temp = machine["Temperature (°C)"]
        vibration = machine["Vibration (mm/s)"]

        def reading_cell(value, threshold):
            if pd.isna(value):
                return '<td><span class="no-data">No data</span></td>'
            value_text = f"{float(value):.1f}"
            highlight = ' class="reading-alert"' if float(value) > threshold else ""
            return f"<td{highlight}>{value_text}</td>"

        table_rows.append(
            "<tr>"
            f'<td>{html_escape(str(machine["Machine"]))}</td>'
            + reading_cell(temp, TEMP_THRESHOLD)
            + reading_cell(vibration, VIBRATION_THRESHOLD)
            + f'<td><span class="status-{status_class}">'
              '<span class="status-dot"></span>'
              f'{html_escape(status)}</span></td>'
            + f'<td>{html_escape(str(machine["Possible faults"]))}</td>'
            + "</tr>"
        )

    table_html = (
        '<div class="fleet-table-wrap"><table class="showcase-table">'
        '<thead><tr><th>Machine</th><th>Temperature (°C)</th>'
        '<th>Vibration (mm/s)</th><th>Status</th>'
        '<th>Possible faults</th></tr></thead><tbody>'
        + "".join(table_rows) + '</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.caption(
        "Status is determined by demonstration sensor thresholds, "
        "not by a trained predictive maintenance model."
    )


# ===================================
# Machine Details
# ===================================

with tab_machine:
    st.subheader(f"Machine {selected_machine}")

    sensors = retrieve_sensors(selected_machine)
    maintenance = retrieve_maintenance(selected_machine)

    temp = sensors.get("temperature")
    vibration = sensors.get("vibration")

    if temp is None or vibration is None:
        status = "Unknown"
    elif temp > TEMP_THRESHOLD or vibration > VIBRATION_THRESHOLD:
        status = "Attention"
    else:
        status = "Normal"

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Temperature",
        f"{temp:.1f} °C" if temp is not None else "N/A",
    )
    c2.metric(
        "Vibration",
        f"{vibration:.1f} mm/s"
        if vibration is not None else "N/A",
    )
    c3.metric("Machine Status", status)

    st.divider()
    st.subheader("Anomaly Detection")

    for anomaly in detect_anomalies(sensors):
        if "exceeds" in anomaly.lower():
            st.warning(anomaly)
        elif "unavailable" in anomaly.lower():
            st.info(anomaly)
        else:
            st.success(anomaly)

    st.divider()
    st.subheader("Maintenance Relationships")

    if maintenance:
        st.dataframe(
            pd.DataFrame(maintenance),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No maintenance relationships found.")


# ===================================
# Knowledge Graph
# ===================================

with tab_graph:
    st.subheader(
        f"Knowledge Graph — {selected_machine}"
    )

    st.caption(
        "Explore machine, component, possible fault, "
        "sensor, and maintenance relationships."
    )

    try:
        graph_html = get_graph_html(selected_machine)

        components.html(
            graph_html,
            height=630,
            scrolling=True,
        )

    except Exception as error:
        st.error(f"Graph visualization failed: {error}")

    st.markdown(
        '<div class="graph-legend">'
        '<span><i class="graph-dot" style="background:#22D3EE"></i>Machine</span>'
        '<span><i class="graph-dot" style="background:#60A5FA"></i>Component</span>'
        '<span><i class="graph-dot" style="background:#A78BFA"></i>Fault / entity</span>'
        '<span><i class="graph-dot" style="background:#FBBF24"></i>Reading / action</span>'
        '</div>', unsafe_allow_html=True,
    )

    st.info(
        "The graph includes owl:sameAs links between "
        "the two representations of selected machines."
    )


# ===================================
# AI Assistant
# ===================================

with tab_ai:
    st.subheader("AI Maintenance Assistant")

    st.caption(
        "Hybrid retrieval: sensor data + SPARQL graph facts "
        "+ semantic document search."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_machine" not in st.session_state:
        st.session_state.chat_machine = selected_machine

    if st.session_state.chat_machine != selected_machine:
        st.session_state.chat_history = []
        st.session_state.chat_machine = selected_machine

    def render_answer_with_evidence(answer_text):
        st.markdown(answer_text)
        labels = sorted(set(re.findall(r"\\[(SENSOR|KG|DOC)-(\\d+)\\]", answer_text)))
        if labels:
            pills = []
            for kind, number in labels:
                css_kind = "graph" if kind == "KG" else "doc" if kind == "DOC" else ""
                pills.append(
                    f'<span class="evidence-pill {css_kind}">'
                    f'{html_escape(kind)}-{html_escape(number)}</span>'
                )
            st.markdown(
                '<div class="evidence-strip"><strong>Evidence references</strong> '
                + "".join(pills) + '</div>', unsafe_allow_html=True,
            )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            (render_answer_with_evidence(message["content"])
             if message["role"] == "assistant" else st.markdown(message["content"]))

    question = st.chat_input(
        f"Ask about machine {selected_machine}..."
    )

    if question:
        # Explicitly scope questions to the selected machine
        if selected_machine.lower() not in question.lower():
            full_question = (
                f"For machine {selected_machine}: {question}"
            )
        else:
            full_question = question

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing retrieved evidence..."):
                try:
                    answer = answer_with_ollama(full_question)
                    render_answer_with_evidence(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except Exception as error:
                    st.error(f"Analysis failed: {error}")


# ===================================
# Evidence Explorer
# ===================================

with tab_evidence:
    st.subheader(
        f"Evidence Explorer — {selected_machine}"
    )

    evidence_question = st.text_input(
        "Retrieval question",
        value=(
            f"What could cause abnormal temperature "
            f"and vibration in {selected_machine}?"
        ),
    )

    if st.button("Retrieve Evidence", type="primary"):
        with st.spinner("Retrieving graph and documents..."):
            try:
                context = build_hybrid_context(
                    selected_machine,
                    evidence_question,
                )

                sensor_facts = retrieve_sensors(selected_machine)
                maintenance_facts = retrieve_maintenance(selected_machine)
                preview_docs = retrieve_documents(evidence_question, top_k=1)

                temperature = sensor_facts.get("temperature")
                vibration = sensor_facts.get("vibration")
                temp_text = f"{temperature:.1f} °C" if temperature is not None else "No data"
                vib_text = f"{vibration:.1f} mm/s" if vibration is not None else "No data"
                first_fact = maintenance_facts[0] if maintenance_facts else {}
                fault_text = html_escape(str(first_fact.get("fault", "Not documented")))
                component_text = html_escape(str(first_fact.get("component", "Not documented")))
                doc_source = html_escape(str(preview_docs[0]["source"])) if preview_docs else "No document"
                doc_excerpt = html_escape(str(preview_docs[0]["content"])[:190]) if preview_docs else "No matching document found."

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown(
                        f'<div class="evidence-card sensor"><h4><span class="evidence-icon">{ui_icon("sensor")}</span> Sensor Evidence</h4><div class="evidence-body">'
                        f'<p><strong>Machine:</strong> {selected_machine}</p>'
                        f'<p><strong>Temperature:</strong> {temp_text}</p>'
                        f'<p><strong>Vibration:</strong> {vib_text}</p>'
                        f'<p><strong>Thresholds:</strong> {TEMP_THRESHOLD} °C / {VIBRATION_THRESHOLD} mm/s</p></div></div>',
                        unsafe_allow_html=True,
                    )
                with ec2:
                    st.markdown(
                        f'<div class="evidence-card graph"><h4><span class="evidence-icon">{ui_icon("graph")}</span> Knowledge Graph Facts</h4><div class="evidence-body">'
                        f'<p><strong>Component:</strong> {component_text}</p>'
                        f'<p><strong>Possible fault:</strong> {fault_text}</p>'
                        f'<p><strong>Machine:</strong> {selected_machine}</p></div></div>',
                        unsafe_allow_html=True,
                    )
                with ec3:
                    st.markdown(
                        f'<div class="evidence-card docs"><h4><span class="evidence-icon">{ui_icon("file")}</span> Retrieved Document</h4><div class="evidence-body">'
                        f'<p><strong>Source:</strong> {doc_source}</p>'
                        f'<p>{doc_excerpt}</p></div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Combined Evidence")
                st.code(context, language="text")

                docs = retrieve_documents(
                    evidence_question,
                    top_k=2,
                )

                st.markdown("#### Retrieved Documents")

                for i, doc in enumerate(docs, start=1):
                    with st.expander(
                        f"DOC-{i}: {doc['source']} "
                        f"(similarity: {doc['score']:.3f})"
                    ):
                        st.write(doc["content"])

            except Exception as error:
                st.error(f"Retrieval failed: {error}")


# -----------------------------------
# Footer
# -----------------------------------

st.divider()
st.caption(
    "Industrial GraphRAG Copilot · "
    "Synthetic data · Local LLM · Explainable retrieval"
)

