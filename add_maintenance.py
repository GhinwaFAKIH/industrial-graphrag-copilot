
from rdflib import Graph, Namespace, RDF, Literal

EX = Namespace("http://example.org/industrial/")

# Load the graph created in Step 1
g = Graph()
g.parse("industrial_kg.ttl", format="turtle")

# Add industrial components
g.add((EX.M102, EX.hasComponent, EX.Bearing102))
g.add((EX.M104, EX.hasComponent, EX.CoolingFan104))

# Add possible faults
g.add((EX.Bearing102, EX.mayHaveFault, EX.BearingWear))
g.add((EX.CoolingFan104, EX.mayHaveFault, EX.CoolingFailure))

# Add maintenance recommendations
g.add((
    EX.BearingWear,
    EX.recommendedAction,
    Literal("Inspect bearing lubrication and alignment")
))

g.add((
    EX.CoolingFailure,
    EX.recommendedAction,
    Literal("Inspect cooling fan and airflow")
))

# Save enriched graph
g.serialize("industrial_kg_enriched.ttl", format="turtle")

print(f"Enriched graph contains {len(g)} triples")

# Query possible faults and actions
query = """
PREFIX ex: <http://example.org/industrial/>

SELECT ?machine ?component ?fault ?action
WHERE {
    ?machine ex:hasComponent ?component .
    ?component ex:mayHaveFault ?fault .
    ?fault ex:recommendedAction ?action .
}
"""

for row in g.query(query):
    print(
        f"Machine: {row.machine.split('/')[-1]}\n"
        f"Component: {row.component.split('/')[-1]}\n"
        f"Possible fault: {row.fault.split('/')[-1]}\n"
        f"Recommended action: {row.action}\n"
    )

