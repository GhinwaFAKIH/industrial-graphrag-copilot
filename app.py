
from pathlib import Path

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
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #101B2E;
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
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Helpers
# -----------------------------------

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

    # Compact, readable graph layout for portfolio screenshots.
    net.set_options("""
    {
      "nodes": {
        "shape": "dot", "size": 23, "borderWidth": 2,
        "font": {"size": 19, "color": "#F8FAFC",
                 "strokeWidth": 3, "strokeColor": "#101827"}
      },
      "edges": {
        "color": {"color": "#64748B", "highlight": "#22D3EE"},
        "width": 2,
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
        "font": {"size": 14, "color": "#CBD5E1",
                 "strokeWidth": 3, "strokeColor": "#101827"},
        "smooth": false
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -38,
          "centralGravity": 0.04,
          "springLength": 95,
          "springConstant": 0.12,
          "damping": 0.85
        },
        "stabilization": {"enabled": true, "iterations": 250}
      },
      "interaction": {"hover": true, "navigationButtons": true,
                      "keyboard": true, "zoomView": true}
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
                label=str(o),
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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Machines", len(MACHINES))
    c2.metric("Attention Required", attention_count)
    c3.metric("Normal", normal_count)
    c4.metric("Unknown", unknown_count)

    st.divider()

    display_df = fleet_df.fillna("No data")
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )

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

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
                    st.markdown(answer)

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

