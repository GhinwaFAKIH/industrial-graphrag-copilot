"""Public Streamlit demo with evidence-grounded Ollama Cloud answers."""
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from ollama_cloud import ask_ollama
from graph_visualization import show_graph
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
st.info("Public evaluation prototype · non-confidential data only · not a diagnostic or safety system. Answers use Ollama Cloud and retrieved evidence.")
source = st.radio("Choose how to explore", ["Try demo data", "Upload your own knowledge graph"], horizontal=True, key="demo_source")
if source == "Try demo data":
    machine = st.selectbox("Select a simulated machine", [f"M10{i}" for i in range(1, 6)], index=1)
    fleet = []
    for mid in [f"M10{i}" for i in range(1, 6)]:
        s = sensors(mid)
        flag = (s.get("temperature", 0) > TEMP_LIMIT or s.get("vibration", 0) > VIB_LIMIT) if s else None
        fleet.append({"Machine": mid, "Temperature °C": s.get("temperature"), "Vibration mm/s": s.get("vibration"), "Demo status": "Attention" if flag else "Normal" if flag is False else "Unknown"})
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fleet Overview", "Machine Details", "Knowledge Graph", "AI Assistant", "Evidence Explorer"])
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
    with tab4:
        question = st.text_input("Your question", value=f"What might explain abnormal temperature and vibration in {machine}?")
        if st.button("Ask Copilot", type="primary"):
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
            evidence = "\n".join(
                ([f"[SENSOR-1] {machine}: {s}"] if s else [])
                + [f"[KG-{i}] {fact}" for i, fact in enumerate(facts, 1)]
                + [f"[DOC-{i}] {name}: {body[:1500]}" for i, (_, name, body) in enumerate(docs_for(question), 1)]
            )
            with st.spinner("Asking Ollama Cloud..."):
                answer, error = ask_ollama(question, evidence)
            if error:
                st.warning(error)
            elif answer:
                st.subheader("Copilot answer")
                st.write(answer)
            st.caption("Verify the supporting evidence. Document retrieval uses keyword overlap.")

    with tab3:
        st.subheader("Interactive Knowledge Graph")
        focus = EX[machine]
        show_graph(g, focus=focus)
        st.subheader("RDF triples")
        st.caption("Drag nodes, zoom, and hover to explore relationships.")
    with tab5:
        st.subheader("Evidence Explorer")
        st.subheader("Sample RDF knowledge graph")
        st.metric("Triples", len(g))
        st.dataframe(pd.DataFrame([(str(x), str(y), str(z)) for x,y,z in list(g)[:100]], columns=["Subject","Predicate","Object"]), use_container_width=True, hide_index=True)

else:

    st.warning("Public demo: upload only synthetic or non-confidential RDF/Turtle data. This is not a secure enterprise upload service.")
    st.caption("Turtle (.ttl) only · maximum 2 MB · maximum 20,000 triples. No files intentionally saved to disk.")
    upload = st.file_uploader("Upload RDF/Turtle file", type=["ttl"], key="public_rdf_upload")
    if upload is None:
        st.info("Choose a .ttl file to unlock the four exploration tabs below.")
    elif upload.size > 2 * 1024 * 1024:
        st.error("File exceeds the 2 MB demo limit.")
    else:
        try:
            from io import BytesIO
            from rdflib.namespace import RDF
            user_graph = Graph()
            user_graph.parse(source=BytesIO(upload.getvalue()), format="turtle", publicID="urn:public-demo:upload:")
            if len(user_graph) > 20000:
                st.error("Graph exceeds the 20,000-triple demo limit.")
            else:
                def compact(term):
                    return user_graph.namespace_manager.normalizeUri(term) if isinstance(term, URIRef) else str(term)
                triples = sorted([(compact(a), compact(b), compact(c)) for a,b,c in user_graph])
                predicates = sorted(set(row[1] for row in triples))
                t1,t2,t3,t4,t5 = st.tabs(["Fleet Overview", "Machine Details", "Knowledge Graph", "AI Assistant", "Evidence Explorer"])
                with t1:
                    a,b,c = st.columns(3)
                    a.metric("Triples", len(user_graph))
                    b.metric("Subjects", len(set(user_graph.subjects())))
                    c.metric("Predicates", len(set(user_graph.predicates())))
                    st.caption("Overview of the uploaded RDF graph. No sample industrial data is mixed in.")
                with t2:
                    types = sorted([(compact(x),compact(y)) for x,y in user_graph.subject_objects(RDF.type)])
                    st.subheader("Machine Details — RDF entities and types")
                    if types:
                        st.dataframe(pd.DataFrame(types[:200],columns=["Entity","Type"]),hide_index=True,use_container_width=True)
                    else:
                        st.info("No rdf:type statements found.")
                with t4:
                    st.subheader("Ask your RDF knowledge graph")
                    question = st.text_input("Ask a question about entities or relationships", key="own_graph_query")
                    if st.button("Ask Copilot", type="primary", key="own_graph_retrieve"):
                        words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
                        ranked = sorted(triples, key=lambda row: -sum(any(w in value.lower() for value in row) for w in words))
                        results = [row for row in ranked if any(any(w in value.lower() for value in row) for w in words)][:35]
                        if not results:
                            st.warning("No matching RDF evidence. Try an entity or predicate name from your graph.")
                        else:
                            evidence = "\n".join(f"[KG-{i}] {a} {b} {c}" for i, (a, b, c) in enumerate(results, 1))
                            with st.spinner("Asking Ollama Cloud..."):
                                answer, error = ask_ollama(question, evidence)
                            if error:
                                st.warning(error)
                            elif answer:
                                st.subheader("Copilot answer")
                                st.write(answer)
                            with st.expander("Supporting RDF triples"):
                                st.dataframe(pd.DataFrame(results, columns=["Subject", "Predicate", "Object"]), hide_index=True, use_container_width=True)
                            st.caption("Uploaded-graph retrieval is lexical, not full natural-language-to-SPARQL reasoning.")
                with t3:
                    st.subheader("Interactive Knowledge Graph")
                    st.caption("Drag nodes and zoom. A bounded sample is shown for large uploads.")
                    show_graph(user_graph)
                with t5:
                    st.subheader("Evidence Explorer — RDF triples")
                    chosen = st.selectbox("Predicate",["All predicates"]+predicates,key="own_predicate")
                    search = st.text_input("Filter URI or literal",key="own_filter").lower().strip()
                    rows = [row for row in triples if (chosen=="All predicates" or row[1]==chosen) and (not search or any(search in value.lower() for value in row))]
                    st.dataframe(pd.DataFrame(rows[:200],columns=["Subject","Predicate","Object"]),hide_index=True,use_container_width=True)
                    st.caption(f"Showing {min(200,len(rows))} of {len(rows)} matching triples.")
        except Exception:
            st.error("Could not parse the Turtle file. Please check its RDF syntax.")

st.divider()
st.markdown("Interested in evaluating this approach with your use case? **[Contact by email](mailto:ghinwafkh99@gmail.com?subject=Industrial%20GraphRAG%20Copilot%20feedback)**")
