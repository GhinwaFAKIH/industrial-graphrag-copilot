"""Public, no-LLM Streamlit demo with synthetic RDF evidence only."""
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from rdflib import Graph, Namespace

st.set_page_config(page_title="Industrial GraphRAG Copilot — Public Demo", page_icon="🏭", layout="wide")
BASE = Path(__file__).resolve().parent
EX = Namespace("http://example.org/industrial/")
SENSOR = Namespace("https://example.org/industrial/")
TEMP_LIMIT = 80.0
VIB_LIMIT = 5.0

@st.cache_resource
def load_graph():
    graph = Graph()
    graph.parse(str(BASE / "industrial_kg_aligned.ttl"), format="turtle")
    return graph

g = load_graph()

def sensors(machine):
    query = """PREFIX s: <https://example.org/industrial/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?temperature ?vibration WHERE {
      ?sensorMachine owl:sameAs ?maintenanceMachine .
      ?sensorMachine s:temperatureC ?temperature .
      ?sensorMachine s:vibrationMmS ?vibration .
    }"""
    for row in g.query(query, initBindings={"maintenanceMachine": EX[machine]}):
        return {"temperature": float(row.temperature), "vibration": float(row.vibration)}
    return {}

def maintenance(machine):
    query = """PREFIX ex: <http://example.org/industrial/>
    SELECT ?component ?fault ?action WHERE {
      ?machine ex:hasComponent ?component .
      ?component ex:mayHaveFault ?fault .
      ?fault ex:recommendedAction ?action .
    }"""
    return [{"component": str(r.component).split("/")[-1],
             "fault": str(r.fault).split("/")[-1],
             "action": str(r.action)} for r in g.query(query, initBindings={"machine": EX[machine]})]

def docs_for(question, limit=2):
    words = set(re.findall(r"[a-z]{3,}", question.lower()))
    results = []
    for p in sorted((BASE / "maintenance_docs").glob("*.txt")):
        body = p.read_text(encoding="utf-8")
        terms = set(re.findall(r"[a-z]{3,}", (p.stem + " " + body).lower()))
        results.append((len(words & terms), p.name, body))
    return sorted(results, key=lambda item: (-item[0], item[1]))[:limit]

st.title("🏭 Industrial GraphRAG Copilot")
st.info("Public evaluation demo · synthetic data only · not a diagnostic or safety system. This cloud edition provides deterministic evidence retrieval, not LLM-generated answers.")
machine = st.selectbox("Select a simulated machine", [f"M10{i}" for i in range(1, 6)], index=1)
fleet = []
for mid in [f"M10{i}" for i in range(1, 6)]:
    s = sensors(mid)
    flag = (s.get("temperature", 0) > TEMP_LIMIT or s.get("vibration", 0) > VIB_LIMIT) if s else None
    fleet.append({"Machine": mid, "Temperature °C": s.get("temperature"), "Vibration mm/s": s.get("vibration"), "Demo status": "Attention" if flag else "Normal" if flag is False else "Unknown"})
tab1, tab2, tab3 = st.tabs(["Fleet", "Machine evidence", "Ask & retrieve"])
with tab1:
    st.dataframe(pd.DataFrame(fleet), use_container_width=True, hide_index=True)
    st.caption("Illustrative thresholds: temperature > 80 °C or vibration > 5 mm/s.")
with tab2:
    s = sensors(machine)
    facts = maintenance(machine)
    a,b = st.columns(2)
    a.metric("Temperature", f'{s["temperature"]:.1f} °C' if s else "N/A")
    b.metric("Vibration", f'{s["vibration"]:.1f} mm/s' if s else "N/A")
    st.subheader("RDF/SPARQL maintenance facts")
    if facts:
        st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)
    else:
        st.warning("No maintenance facts available for this machine.")
with tab3:
    question = st.text_input("Your question", value=f"What might explain abnormal temperature and vibration in {machine}?")
    if st.button("Retrieve evidence", type="primary"):
        s = sensors(machine)
        facts = maintenance(machine)
        st.subheader("Evidence-based summary")
        if s:
            st.write(f"[SENSOR-1] {machine}: temperature {s['temperature']:.1f} °C; [SENSOR-2] vibration {s['vibration']:.1f} mm/s.")
            if s["temperature"] > TEMP_LIMIT or s["vibration"] > VIB_LIMIT:
                st.warning("One or more illustrative thresholds are exceeded. This does not confirm a fault.")
            else:
                st.success("No illustrative threshold exceedance in this snapshot.")
        else:
            st.warning("No sensor evidence available.")
        for i, fact in enumerate(facts, 1):
            st.write(f"[KG-{i}] Component: {fact['component']}; possible fault: {fact['fault']}; documented action: {fact['action']}")
        if not facts:
            st.write("No maintenance relationships found.")
        st.subheader("Related synthetic maintenance documents")
        for i, (_, filename, body) in enumerate(docs_for(question), 1):
            with st.expander(f"[DOC-{i}] {filename}"):
                st.write(body)
        st.caption("Document matching in this public edition uses keyword overlap. The local project includes semantic embeddings and Ollama; those components are not active here.")
st.divider()
st.markdown("Interested in evaluating this approach with your use case? **[Contact by email](mailto:ghinwafkh99@gmail.com?subject=Industrial%20GraphRAG%20Copilot%20feedback)**")
