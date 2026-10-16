from rdflib import Graph, Namespace, Literal, RDF, XSD

EX = Namespace('https://example.org/industrial/')
g = Graph()
g.bind('ex', EX)

machines = [
    ('M-101', 61, 2.1, 'Bearing'),
    ('M-102', 89, 6.2, 'Bearing'),
    ('M-103', 66, 2.8, 'Motor'),
    ('M-104', 83, 3.1, 'CoolingSystem'),
    ('M-105', 59, 1.7, 'Motor'),
]
for name, temperature, vibration, component in machines:
    machine = EX[name]
    part = EX[f'{name}-{component}']
    g.add((machine, RDF.type, EX.Machine))
    g.add((machine, EX.temperatureC, Literal(temperature, datatype=XSD.decimal)))
    g.add((machine, EX.vibrationMmS, Literal(vibration, datatype=XSD.decimal)))
    g.add((machine, EX.hasComponent, part))
    g.add((part, RDF.type, EX[component]))

# Demonstration-only thresholds, not engineering safety limits.
query = '''PREFIX ex: <https://example.org/industrial/>
SELECT ?machine ?temperature ?vibration WHERE {
  ?machine a ex:Machine ; ex:temperatureC ?temperature ; ex:vibrationMmS ?vibration .
  FILTER (?temperature > 80 || ?vibration > 5)
} ORDER BY ?machine'''

print('Machines exceeding demo thresholds (80°C or 5 mm/s):')
for machine, temp, vib in g.query(query):
    print(f'- {str(machine).split("/")[-1]}: {temp}°C, {vib} mm/s')

g.serialize('industrial_kg.ttl', format='turtle')
print(f'\nSaved industrial_kg.ttl ({len(g)} triples)')
