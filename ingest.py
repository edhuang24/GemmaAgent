# Path provides a clean, cross-platform way to work with file system paths
from pathlib import Path

# partition auto-detects file type (PDF, TXT, MD, etc.) and extracts text consistently
from unstructured.partition.auto import partition

def load_documents(folder_path: str) -> list[dict]:
    docs = []

    # Iterate over every item in the given folder
    for file_path in Path(folder_path).iterdir():

        # Skip subdirectories — only process files
        if file_path.is_file():

            # partition() handles format detection automatically — same call works for all file types
            elements = partition(filename=str(file_path))

            # Each file is returned as a list of elements (paragraphs, titles, etc.)
            # We filter out empty elements and join everything into one string per document
            full_text = "\n".join([el.text for el in elements if el.text])
            
            # Store the text alongside metadata so we know which file each chunk came from later
            docs.append({
                "text": full_text,
                "metadata": {"filename": file_path.name, "path": str(file_path)}
            })
    return docs

if __name__ == "__main__":
    docs = load_documents("docs/")
    # Print a summary of each loaded document to verify everything parsed correctly
    for doc in docs:
        print(doc["metadata"]["filename"], "—", len(doc["text"]), "chars")
