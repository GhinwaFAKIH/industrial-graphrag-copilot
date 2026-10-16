
from rdflib import Graph, URIRef, Literal
from pyvis.network import Network

GRAPH_FILE = "industrial_kg_aligned.ttl"

g = Graph()
g.parse(GRAPH_FILE, format="turtle")

net = Network(
    height="600px",
    width="100%",
    directed=True,
    bgcolor="#101827",
    font_color="white",
    cdn_resources="in_line"
)

net.barnes_hut()

def short_name(uri):
    return str(uri).split("/")[-1].split("#")[-1]

for subject, predicate, obj in g:

    source = str(subject)
    relation = short_name(predicate)

    if isinstance(subject, URIRef):
        net.add_node(
            source,
            label=short_name(subject),
            color="#38bdf8",
            title=source
        )

    if isinstance(obj, URIRef):
        target = str(obj)

        net.add_node(
            target,
            label=short_name(obj),
            color="#a78bfa",
            title=target
        )

        net.add_edge(
            source,
            target,
            title=relation,
            label=relation
        )

    elif isinstance(obj, Literal):
        target = f"{source}:{relation}:{obj}"

        net.add_node(
            target,
            label=str(obj),
            color="#fbbf24",
            title=f"{relation}: {obj}"
        )

        net.add_edge(
            source,
            target,
            title=relation,
            label=relation
        )

net.write_html("knowledge_graph.html")

print("Knowledge Graph visualization generated.")

