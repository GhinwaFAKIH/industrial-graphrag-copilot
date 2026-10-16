
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS_DIR = Path("maintenance_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append({
                "source": file_path.name,
                "content": content
            })

    return documents


documents = load_documents()

if documents:
    embeddings = model.encode(
        [doc["content"] for doc in documents],
        normalize_embeddings=True
    )
else:
    embeddings = np.empty((0, 384))


def retrieve_documents(query, top_k=2):
    if not documents:
        return []

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        doc = documents[int(index)]

        results.append({
            "source": doc["source"],
            "content": doc["content"],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":
    question = "What should I inspect if a bearing is overheating?"

    for result in retrieve_documents(question):
        print(f"\nSource: {result['source']}")
        print(f"Similarity: {result['score']:.3f}")
        print(result["content"])

