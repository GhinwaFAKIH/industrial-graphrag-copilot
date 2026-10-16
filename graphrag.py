
import re
from rdflib import Graph, Namespace

EX = Namespace("http://example.org/industrial/")

# Load enriched Knowledge Graph
g = Graph()
g.parse("industrial_kg_enriched.ttl", format="turtle")


def extract_machine(question):
    """Extract machine ID from the user's question."""
    match = re.search(r"\bM-?10[1-5]\b", question, re.I)
    if match:
        return match.group().upper().replace("-", "")
    return None


def retrieve_from_graph(machine_id):
    """Retrieve possible faults and maintenance actions."""
    query = """
    PREFIX ex: <http://example.org/industrial/>

    SELECT ?component ?fault ?action
    WHERE {
        ?machine ex:hasComponent ?component .
        ?component ex:mayHaveFault ?fault .
        ?fault ex:recommendedAction ?action .
    }
    """

    results = []

    for row in g.query(query, initBindings={"machine": EX[machine_id]}):
        results.append({
            "component": str(row.component).split("/")[-1],
            "fault": str(row.fault).split("/")[-1],
            "action": str(row.action)
        })

    return results


def answer_question(question):
    machine_id = extract_machine(question)

    if not machine_id:
        return "Please specify a machine, e.g. M102."

    facts = retrieve_from_graph(machine_id)

    if not facts:
        return f"No maintenance information found for {machine_id}."

    answer = f"Analysis for machine {machine_id}:\n\n"

    for fact in facts:
        answer += (
            f"Component: {fact['component']}\n"
            f"Possible fault: {fact['fault']}\n"
            f"Recommended action: {fact['action']}\n\n"
        )

    return answer
    
 
def retrieve_sensor_data(machine_id):
    """Retrieve sensor measurements from the RDF graph."""

    query = """
    PREFIX ex: <http://example.org/industrial/>

    SELECT ?predicate ?value
    WHERE {
        ?machine ?predicate ?value .
        FILTER(
            ?predicate IN (
                ex:hasTemperature,
                ex:hasVibration
            )
        )
    }
    """

    results = {}

    for row in g.query(
        query,
        initBindings={"machine": EX[machine_id]}
    ):
        sensor_name = str(row.predicate).split("/")[-1]
        results[sensor_name] = float(row.value)

    return results



if __name__ == "__main__":
    print("Industrial GraphRAG Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        print("\n" + answer_question(question))
        
        

