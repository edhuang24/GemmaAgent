from sentence_transformers import SentenceTransformer
import chromadb

def retrieve(query: str, n_results: int = 5, collection_name: str = "knowledge_base") -> list[dict]:
    # Load the same embedding model used during ingest
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to the same persistent ChromaDB instance created during ingest
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)

    # Embed the query into the same vector space as the stored chunks
    query_embedding = model.encode([query]).tolist()

    # Search for the top-k most similar chunks by vector distance
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    # Repack into a clean list of dicts for easy consumption downstream
    chunks = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, "metadata": metadata})
    return chunks

if __name__ == "__main__":
    query = "What did Darwin say about natural selection?"
    results = retrieve(query)
    for i, chunk in enumerate(results):
        print(f"\n--- Result {i+1} ({chunk['metadata']['filename']}) ---")
        print(chunk["text"][:300])