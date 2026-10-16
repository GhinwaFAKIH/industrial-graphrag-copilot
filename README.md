# Industrial GraphRAG Copilot — Day 1 starter

A tiny synthetic industrial knowledge graph with five machines and SPARQL anomaly filtering.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

Outputs `industrial_kg.ttl` and prints machines exceeding demonstration thresholds. All data and thresholds are fictional, not validated industrial diagnostics.

## Next

Add maintenance logs, fault relationships, and evidence-backed graph retrieval for the copilot.
