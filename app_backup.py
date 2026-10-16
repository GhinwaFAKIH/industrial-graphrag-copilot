
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphrag_ollama import (
    retrieve_sensors,
    retrieve_maintenance,
    detect_anomalies,
    build_context,
    build_hybrid_context,
    answer_with_ollama,
    TEMP_THRESHOLD,
    VIBRATION_THRESHOLD,
)

# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="Industrial GraphRAG Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------
# Header
# -----------------------------------------

st.title("🏭 Industrial GraphRAG Copilot")

st.caption(
    "Industrial Equipment Monitoring | "
    "RDF Knowledge Graphs | SPARQL | Ollama"
)

st.info(
    "This prototype uses simulated industrial data. "
    "Anomaly thresholds are illustrative and do not "
    "represent certified industrial safety limits."
)

# -----------------------------------------
# Sidebar
# -----------------------------------------

st.sidebar.title("Control Panel")

machines = [f"M10{i}" for i in range(1, 6)]

selected_machine = st.sidebar.selectbox(
    "Select Machine",
    machines,
    index=1,
)

st.sidebar.divider()

st.sidebar.subheader("Technology Stack")
st.sidebar.markdown(
    """
    - Python / RDFLib
    - RDF / SPARQL
    - Knowledge Graphs
    - Ollama / Gemma 3
    - Streamlit
    - PyVis
    """
)

st.sidebar.divider()
st.sidebar.caption("Industrial AI Demonstration")

# -----------------------------------------
# Retrieve machine data
# -----------------------------------------

sensors = retrieve_sensors(selected_machine)
maintenance = retrieve_maintenance(selected_machine)
anomalies = detect_anomalies(sensors)

temperature = sensors.get("temperature")
vibration = sensors.get("vibration")

has_temperature = temperature is not None
has_vibration = vibration is not None

temperature_alert = (
    has_temperature and temperature > TEMP_THRESHOLD
)
vibration_alert = (
    has_vibration and vibration > VIBRATION_THRESHOLD
)

if not has_temperature or not has_vibration:
    machine_status = "Unknown"
elif temperature_alert or vibration_alert:
    machine_status = "Attention"
else:
    machine_status = "Normal"

# -----------------------------------------
# Tabs
# -----------------------------------------

tab_overview, tab_graph, tab_ai = st.tabs(
    [
        "📊 Machine Monitoring",
        "🕸️ Knowledge Graph",
        "🤖 AI Maintenance Assistant",
    ]
)

# =========================================
# TAB 1: MACHINE MONITORING
# =========================================

with tab_overview:

    st.header(f"Machine {selected_machine}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Temperature",
            f"{temperature:.1f} °C"
            if has_temperature else "N/A",
        )

    with col2:
        st.metric(
            "Vibration",
            f"{vibration:.1f} mm/s"
            if has_vibration else "N/A",
        )

    with col3:
        st.metric(
            "Machine Status",
            machine_status,
        )

    st.divider()

    # Sensor data
    st.subheader("Sensor Measurements")

    sensor_rows = []

    if has_temperature:
        sensor_rows.append({
            "Sensor": "Temperature",
            "Value": temperature,
            "Unit": "°C",
            "Demo Threshold": TEMP_THRESHOLD,
            "Status": (
                "Above threshold"
                if temperature_alert else "Within threshold"
            ),
        })

    if has_vibration:
        sensor_rows.append({
            "Sensor": "Vibration",
            "Value": vibration,
            "Unit": "mm/s",
            "Demo Threshold": VIBRATION_THRESHOLD,
            "Status": (
                "Above threshold"
                if vibration_alert else "Within threshold"
            ),
        })

    if sensor_rows:
        st.dataframe(
            pd.DataFrame(sensor_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No sensor measurements available.")

    # Anomaly detection
    st.divider()
    st.subheader("Anomaly Detection")

    for anomaly in anomalies:
        if "exceeds" in anomaly.lower():
            st.warning(anomaly)
        elif "unavailable" in anomaly.lower():
            st.info(anomaly)
        else:
            st.success(anomaly)

    # Maintenance evidence
    st.divider()
    st.subheader("Maintenance Knowledge")

    if maintenance:
        maintenance_df = pd.DataFrame(maintenance)
        maintenance_df = maintenance_df.rename(
            columns={
                "component": "Component",
                "fault": "Possible Fault",
                "action": "Recommended Action",
            }
        )

        st.dataframe(
            maintenance_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No maintenance recommendations found "
            "for this machine."
        )

# =========================================
# TAB 2: KNOWLEDGE GRAPH
# =========================================

with tab_graph:

    st.header("Interactive Knowledge Graph")

    st.write(
        "Explore machines, components, possible faults, "
        "sensor values, and maintenance recommendations "
        "represented as RDF triples."
    )

    st.caption(
        "Drag nodes to rearrange the graph. "
        "Scroll to zoom and hover to inspect relationships."
    )

    graph_file = Path("knowledge_graph.html")

    if graph_file.exists():

        graph_html = graph_file.read_text(
            encoding="utf-8"
        )

        components.html(
            graph_html,
            height=650,
            scrolling=True,
        )

        st.download_button(
            label="Download Knowledge Graph HTML",
            data=graph_html,
            file_name="knowledge_graph.html",
            mime="text/html",
        )

    else:
        st.warning(
            "Graph visualization not found. "
            "Run `python visualize_graph.py` first."
        )

    st.divider()

    st.subheader("Retrieved RDF Evidence")

    context = build_context(selected_machine)

    if context:
        st.code(context, language="text")
    else:
        st.info(
            "No RDF evidence available for this machine."
        )

    st.caption(
        "Note: the current RDF dataset contains two "
        "URI representations for some machines. "
        "Identity alignment is planned for a later version."
    )

# =========================================
# TAB 3: AI ASSISTANT
# =========================================

with tab_ai:

    st.header("AI Maintenance Assistant")

    st.write(
        "Ask questions about equipment conditions, "
        "sensor anomalies, possible component faults, "
        "and maintenance recommendations."
    )

    st.caption(
        "Answers are generated locally with Ollama "
        "using evidence retrieved from the RDF graph."
    )

    default_question = (
        f"Why might {selected_machine} be malfunctioning? "
        "Analyze its temperature, vibration, "
        "and possible faults."
    )

    question = st.text_area(
        "Your question",
        value=default_question,
        height=110,
        key=f"question_{selected_machine}",
    )

    if st.button(
        "Analyze Machine",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner(
                "Retrieving Knowledge Graph evidence "
                "and generating an explanation..."
            ):
                try:
                    answer = answer_with_ollama(question)

                    st.subheader("AI Analysis")
                    st.markdown(answer)

                    st.divider()
                    st.subheader("Supporting Evidence")

                    evidence = build_hybrid_context(
    	            selected_machine,
                     question
                      )

                    if evidence:
                        with st.expander(
                            "View retrieved RDF evidence",
                            expanded=True,
                        ):
                            st.code(
                                evidence,
                                language="text",
                            )

                except Exception as error:
                    st.error(
                        f"Unable to generate analysis: {error}"
                    )

    st.divider()

    st.markdown("#### Example Questions")

    st.markdown(
        f"""
        - Why might {selected_machine} be malfunctioning?
        - What sensor readings are abnormal for {selected_machine}?
        - What maintenance actions are recommended for {selected_machine}?
        - Which component might be responsible for the issue?
        """
    )

    st.warning(
        "AI-generated explanations are diagnostic suggestions, "
        "not confirmed root-cause analyses. "
        "Human inspection is required."
    )

# -----------------------------------------
# Footer
# -----------------------------------------

st.divider()

st.caption(
    "Industrial GraphRAG Copilot | "
    "Python · RDFLib · SPARQL · Ollama · Streamlit · PyVis"
)

