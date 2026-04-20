from retriever import retrieve

# Each tool is a plain Python function — this is what actually runs when the agent calls the tool
def rag_retrieve(query: str) -> str:
    # Call the retriever and format the results as a single readable string for the LLM
    results = retrieve(query)
    # Join all chunks into one block of text, labelled by source file
    return "\n\n".join([f"[{r['metadata']['filename']}]\n{r['text']}" for r in results])

# The schema is what the LLM sees — it describes the tool's name, purpose, and parameters
# This follows the OpenAI function-calling schema format
RAG_RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        # The name the LLM will use to call this tool
        "name": "rag_retrieve",
        # A clear description so the LLM knows when to use this tool
        "description": "Search the local knowledge base for relevant information. Use this for questions about documents you have ingested.",
        "parameters": {
            # JSON Schema type for the parameters object
            "type": "object",
            "properties": {
                "query": {
                    # The data type of this parameter
                    "type": "string",
                    # Description helps the LLM construct a good query
                    "description": "The search query to look up in the knowledge base"
                }
            },
            # List of parameters the LLM must always provide
            "required": ["query"]
        }
    }
}