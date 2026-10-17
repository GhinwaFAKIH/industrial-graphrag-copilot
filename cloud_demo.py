"""Public, no-LLM Streamlit demo with synthetic RDF evidence only."""
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from rdflib import Graph, Namespace, URIRef, Literal, BNode

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
tab1, tab2, tab3, tab4 = st.tabs(["Fleet", "Machine evidence", "Ask & retrieve", "Upload RDF graph"])
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

with tab4:
    st.subheader("Explore your own RDF knowledge graph")
    st.warning("Public demo: upload only synthetic or non-confidential RDF/Turtle data. Files are processed in this app session; this is not a secure enterprise upload service.")
    st.caption("Turtle (.ttl) only · maximum 2 MB · up to 20,000 triples · no files intentionally written to disk.")
    upload = st.file_uploader("Choose an RDF/Turtle file", type=["ttl"], key="public_rdf_upload")
    if upload is not None:
        raw = upload.getvalue()
        if len(raw) > 2 * 1024 * 1024:
            st.error("File exceeds the 2 MB demo limit.")
        else:
            try:
                from rdflib.parser import create_input_source
                from rdflib.plugins.parsers.notation3 import TurtleParser
                # Parse a local byte stream only. RDFLib's Turtle parser can resolve
                # external references in some contexts; this demo does not accept
                # remote URLs, JSON-LD contexts, or user-supplied SPARQL.
                from io import BytesIO
                user_graph = Graph()
                user_graph.parse(source=BytesIO(raw), format="turtle", publicID="urn:public-demo:upload:")
                if len(user_graph) > 20000:
                    st.error("Graph exceeds the 20,000-triple demo limit.")
                else:
                    subjects = set(user_graph.subjects())
                    predicates = sorted(set(user_graph.predicates()), key=str)
                    objects = set(user_graph.objects())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Triples", len(user_graph))
                    c2.metric("Subjects", len(subjects))
                    c3.metric("Predicates", len(predicates))
                    def compact(term):
                        if isinstance(term, URIRef):
                            try:
                                return user_graph.namespace_manager.normalizeUri(term)
                            except Exception:
                                return str(term)
                        return str(term)
                    if len(user_graph):
                        st.subheader("Explore triples")
                        predicate_options = ["All predicates"] + [str(p) for p in predicates]
                        chosen = st.selectbox("Filter by predicate", predicate_options, key="uploaded_predicate")
                        term_search = st.text_input("Find a URI or literal", placeholder="e.g. Machine, hasComponent, temperature", key="uploaded_search").strip().lower()
                        matching = []
                        for subj, pred, obj in user_graph:
                            if chosen != "All predicates" and str(pred) != chosen:
                                continue
                            values = (compact(subj), compact(pred), compact(obj))
                            if term_search and not any(term_search in v.lower() for v in values):
                                continue
                            matching.append(values)
                            if len(matching) >= 201:
                                break
                        matching.sort()
                        st.dataframe(pd.DataFrame(matching[:200], columns=["Subject", "Predicate", "Object"]), hide_index=True, use_container_width=True)
                        st.caption("Showing at most 200 matching triples. This is graph browsing, not natural-language question answering.")
                        st.subheader("RDF types")
                        from rdflib.namespace import RDF
                        type_rows = [(compact(s), compact(o)) for s, o in user_graph.subject_objects(RDF.type)]
                        if type_rows:
                            st.dataframe(pd.DataFrame(sorted(type_rows)[:100], columns=["Entity", "RDF type"]), hide_index=True, use_container_width=True)
                        else:
                            st.info("No rdf:type statements were found.")
                    else:
                        st.info("This RDF file contains no triples.")
            except Exception:
                st.error("Could not parse this Turtle file. Check that it contains valid RDF/Turtle syntax and uses no remote dependencies.")

st.divider()
st.markdown("Interested in evaluating this approach with your use case? **[Contact by email](mailto:ghinwafkh99@gmail.com?subject=Industrial%20GraphRAG%20Copilot%20feedback)**")
