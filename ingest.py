# Path provides a clean, cross-platform way to work with file system paths
from pathlib import Path

# partition auto-detects file type (PDF, TXT, MD, etc.) and extracts text consistently
from unstructured.partition.auto import partition

# RecursiveCharacterTextSplitter splits on natural boundaries (paragraphs → sentences → words)
# before falling back to hard character cuts — imported from the standalone text splitters package
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SentenceTransformer runs the embedding model locally — no external API calls
from sentence_transformers import SentenceTransformer
# chromadb is our local vector store — persists embeddings to disk for reuse
import chromadb

# Walks a folder and loads all files into a list of dicts with text and metadata
def load_documents(folder_path: str) -> list[dict]:
    docs = []

    # Iterate over every item in the given folder
    for file_path in Path(folder_path).iterdir():
        # Skip subdirectories — only process files
        if file_path.is_file():

            # partition() handles format detection automatically — same call works for all file types
            # languages=["eng"] sets English as default — update or remove this if ingesting non-English documents
            elements = partition(filename=str(file_path), languages=["eng"])

            # Each file is returned as a list of elements (paragraphs, titles, etc.)
            # We filter out empty elements and join everything into one string per document
            full_text = "\n".join([el.text for el in elements if el.text])

            # Store the text alongside metadata so we know which file each chunk came from later
            docs.append({
                "text": full_text,
                "metadata": {"filename": file_path.name, "path": str(file_path)}
            })
    return docs

# Splits each document's text into smaller overlapping chunks for more precise retrieval
def chunk_documents(docs: list[dict]) -> list[dict]:

    # chunk_size: max characters per chunk — large enough for context, small enough to stay focused
    # chunk_overlap: characters shared between adjacent chunks to avoid cutting sentences at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = []
    
    for doc in docs:
        # Split the full document text into a list of smaller string chunks
        splits = splitter.split_text(doc["text"])

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                # Spread the original metadata forward and add chunk_index so we know
                # the position of this chunk within its source document
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks

# Converts chunk text into vectors using the local embedding model
def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    # Load the embedding model — runs locally on CPU/MPS, no API needed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Extract just the text from each chunk to pass to the model
    texts = [chunk["text"] for chunk in chunks]
    # Embed all chunks in one batch — much faster than one at a time
    # .tolist() converts numpy arrays to plain Python lists for ChromaDB compatibility
    return model.encode(texts, show_progress_bar=True).tolist()

# Stores pre-computed embeddings and their source chunks into ChromaDB
def store_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str = "knowledge_base"):
    # Initialize a persistent ChromaDB client — stores the index to disk at ./chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")
    # Get the collection if it exists, or create it if this is the first run
    collection = client.get_or_create_collection(collection_name)

    # Extract parallel lists — ChromaDB requires ids, documents, embeddings, and metadatas
    # to be passed as separate lists of the same length
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    # Build a unique ID per chunk using filename + chunk index to avoid collisions
    ids = [f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_index']}" for chunk in chunks]

    # Store everything in ChromaDB — documents lets us retrieve the original text at query time
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"\nStored {len(chunks)} chunks in ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    # Load raw documents from the docs/ folder
    docs = load_documents("docs/")

    # Print a summary of each loaded document to verify everything parsed correctly
    for doc in docs:
        print(doc["metadata"]["filename"], "—", len(doc["text"]), "chars")

    # Split documents into chunks
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks: {len(chunks)}")

    # Embed chunks and store them in ChromaDB as two explicit steps
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings)