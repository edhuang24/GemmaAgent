from retriever import retrieve

# DDGS is the DuckDuckGo search client — no API key required
from ddgs import DDGS

# subprocess lets us run shell commands from Python and capture their output
import subprocess

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

# Searches the web using DuckDuckGo and returns a summary of the top results
def web_search(query: str) -> str:
    # Open a DuckDuckGo search session
    with DDGS() as ddgs:
        # Fetch the top 5 results — each result has title, href, and body fields
        results = list(ddgs.text(query, max_results=5))
    # Format each result as a readable block for the LLM
    return "\n\n".join([f"[{r['title']}]\n{r['body']}" for r in results])

# Schema describing web_search to the LLM
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        # The name the LLM will use to call this tool
        "name": "web_search",
        # Tells the LLM to use this for current or general knowledge not in the local docs
        "description": "Search the web for current information. Use this for questions not covered by the local knowledge base.",
        "parameters": {
            # JSON Schema type for the parameters object
            "type": "object",
            "properties": {
                "query": {
                    # The data type of this parameter
                    "type": "string",
                    # Description helps the LLM construct a good search query
                    "description": "The search query to look up on the web"
                }
            },
            # The LLM must always provide a query
            "required": ["query"]
        }
    }
}

# Opens the file at the given path and returns its full contents as a string
def read_file(path: str) -> str:
    # "r" mode opens for reading; will raise FileNotFoundError if path doesn't exist
    with open(path, "r") as f:
        return f.read()

# Schema describing read_file to the LLM
READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        # The name the LLM will use to call this tool
        "name": "read_file",
        # Tells the LLM this tool reads local files, not web content
        "description": "Read the contents of a file on the local filesystem.",
        "parameters": {
            # JSON Schema type for the parameters object
            "type": "object",
            "properties": {
                "path": {
                    # The data type of this parameter
                    "type": "string",
                    # Accepts both absolute paths (/Users/...) and relative paths (./docs/file.txt)
                    "description": "The absolute or relative path to the file to read"
                }
            },
            # The LLM must always provide a path
            "required": ["path"]
        }
    }
}

# Writes content to a file at the given path, creating or overwriting it
def write_file(path: str, content: str) -> str:
    # "w" mode creates the file if it doesn't exist and overwrites if it does
    with open(path, "w") as f:
        f.write(content)
    # Return a confirmation string so the LLM knows the write succeeded
    return f"Written to {path}"

# Schema describing write_file to the LLM
WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        # The name the LLM will use to call this tool
        "name": "write_file",
        # Tells the LLM this creates or overwrites — important distinction from append
        "description": "Write content to a file on the local filesystem. Creates the file if it does not exist, overwrites if it does.",
        "parameters": {
            # JSON Schema type for the parameters object
            "type": "object",
            "properties": {
                "path": {
                    # The data type of this parameter
                    "type": "string",
                    # Description helps the LLM provide a valid file path
                    "description": "The path to the file to write"
                },
                "content": {
                    # The data type of this parameter
                    "type": "string",
                    # The full text that will be written to the file
                    "description": "The content to write to the file"
                }
            },
            # The LLM must always provide both path and content
            "required": ["path", "content"]
        }
    }
}

# Runs a shell command and returns its combined stdout and stderr as a string
def run_shell(command: str) -> str:
    # shell=True lets us pass the command as a plain string (vs a list of args)
    # capture_output=True redirects stdout/stderr into the result object instead of printing
    # text=True decodes the output as a string instead of raw bytes
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Combine stdout and stderr so the LLM sees both normal output and any error messages
    return result.stdout + result.stderr

# Schema describing run_shell to the LLM
RUN_SHELL_SCHEMA = {
    "type": "function",
    "function": {
        # The name the LLM will use to call this tool
        "name": "run_shell",
        # Tells the LLM this can run arbitrary shell commands — use sparingly
        "description": "Run a shell command and return its output. Use for file system operations, running scripts, or checking system state.",
        "parameters": {
            # JSON Schema type for the parameters object
            "type": "object",
            "properties": {
                "command": {
                    # The data type of this parameter
                    "type": "string",
                    # Any valid shell command, e.g. "ls -la" or "python3 script.py"
                    "description": "The shell command to execute"
                }
            },
            # The LLM must always provide a command
            "required": ["command"]
        }
    }
}