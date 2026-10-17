"""Safe interactive RDF visualization, limited to a small subgraph."""
from html import escape
from rdflib import URIRef, BNode, Literal
from pyvis.network import Network
import streamlit.components.v1 as components

def show_graph(graph, focus=None, max_edges=160):
    if focus is None:
        triples = sorted(graph, key=lambda t: tuple(map(str, t)))[:max_edges]
    else:
        triples = []
        frontier = {focus}
        visited = set()
        for _ in range(3):
            next_nodes = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for triple in graph.triples((node, None, None)):
                    triples.append(triple)
                    if isinstance(triple[2], (URIRef, BNode)):
                        next_nodes.add(triple[2])
            frontier = next_nodes
        triples = triples[:max_edges]
    if not triples:
        return False
    net = Network(height="550px", width="100%", directed=True,
                  bgcolor="#101827", font_color="#F8FAFC", cdn_resources="in_line")
    net.set_options('{"physics":{"enabled":true,"stabilization":{"iterations":120}},"interaction":{"hover":true,"navigationButtons":true},"edges":{"arrows":{"to":{"enabled":true}},"font":{"size":11,"color":"#CBD5E1","strokeWidth":0}}}')
    seen = set()
    for subject, predicate, obj in triples:
        for term in (subject, obj):
            identifier = str(term)
            if identifier in seen:
                continue
            seen.add(identifier)
            label = graph.namespace_manager.normalizeUri(term) if isinstance(term, URIRef) else str(term)
            color = "#22D3EE" if isinstance(term, URIRef) else "#34D399" if isinstance(term, Literal) else "#A78BFA"
            net.add_node(identifier, label=label[-48:], title=escape(str(term)), color=color, size=17 if term == focus else 11)
        relation = graph.namespace_manager.normalizeUri(predicate)
        net.add_edge(str(subject), str(obj), label=relation.split(":")[-1][-28:], title=escape(str(predicate)))
    components.html(net.generate_html(), height=570, scrolling=False)
    return True
