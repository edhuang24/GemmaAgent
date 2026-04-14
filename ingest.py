from pathlib import Path
from unstructured.partition.auto import partition

def load_documents(folder_path: str) -> list[dict]:
    docs = []
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file():
            elements = partition(filename=str(file_path))
            full_text = "\n".join([el.text for el in elements if el.text])
            docs.append({
                "text": full_text,
                "metadata": {"filename": file_path.name, "path": str(file_path)}
            })
    return docs

if __name__ == "__main__":
    docs = load_documents("docs/")
    for doc in docs:
        print(doc["metadata"]["filename"], "—", len(doc["text"]), "chars")