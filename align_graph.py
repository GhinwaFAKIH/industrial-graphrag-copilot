
from rdflib import Graph, Namespace
from rdflib.namespace import OWL

SOURCE = "industrial_kg_enriched.ttl"
OUTPUT = "industrial_kg_aligned.ttl"

g = Graph()
g.parse(SOURCE, format="turtle")

SENSOR = Namespace("https://example.org/industrial/")
MAINTENANCE = Namespace("http://example.org/industrial/")

g.bind("sensor", SENSOR)
g.bind("maintenance", MAINTENANCE)
g.bind("owl", OWL)

# Link identifiers referring to the same demo machine.
# These mappings are defined by our synthetic dataset.
for i in range(1, 6):
    machine_id = f"M10{i}"
    sensor_id = f"M-10{i}"

    sensor_uri = SENSOR[sensor_id]
    maintenance_uri = MAINTENANCE[machine_id]

    # Add identity links only for resources present in the graph.
    if (
        any(g.triples((sensor_uri, None, None)))
        and any(g.triples((maintenance_uri, None, None)))
    ):
        g.add((sensor_uri, OWL.sameAs, maintenance_uri))
        print(f"Aligned {sensor_id} with {machine_id}")

g.serialize(destination=OUTPUT, format="turtle")

print(f"\nSaved {OUTPUT}")
print(f"Total triples: {len(g)}")

