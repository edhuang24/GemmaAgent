# Use the openai client pointed at LM Studio's local endpoint
from openai import OpenAI

# Import all the tool schemas we've defined
from tools import (
    RAG_RETRIEVE_SCHEMA,
    WEB_SEARCH_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RUN_SHELL_SCHEMA,
)

# Point the client at LM Studio instead of the real OpenAI API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Bundle all schemas into a list — this is what gets sent to the LLM
tools = [
    RAG_RETRIEVE_SCHEMA,
    WEB_SEARCH_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RUN_SHELL_SCHEMA,
]

# The query you want to test — swap this out to try different tools
query = "When was Rome founded?"

# stream=True returns an iterator of chunks instead of waiting for the full response
stream = client.chat.completions.create(
    model="gemma",
    # tool_choice="auto" lets the model decide whether to call a tool or answer directly
    tool_choice="auto",
    tools=tools,
    # Cap token generation — Gemma can loop endlessly generating tool calls without this
    max_tokens=512,
    # Enable streaming so thinking prints to the terminal in real time
    stream=True,
    messages=[{"role": "user", "content": query}],
)

print("--- Thinking ---")

# Tool call data arrives in fragments across chunks — accumulate it here
tool_calls = []

# Iterate over each chunk as it arrives from the model
for chunk in stream:
    delta = chunk.choices[0].delta

    # Thinking and plain text content stream in via delta.content — print immediately
    if delta.content:
        print(delta.content, end="", flush=True)

    # Tool call data is split across chunks — each chunk adds a fragment
    if delta.tool_calls:
        for tc in delta.tool_calls:
            # tc.index tells us which tool call this fragment belongs to
            # If it's a new index, start a new entry in our accumulator
            if tc.index >= len(tool_calls):
                tool_calls.append({"name": "", "arguments": ""})
            # Append the name fragment (arrives first, usually in one chunk)
            if tc.function.name:
                tool_calls[tc.index]["name"] += tc.function.name
            # Append the arguments fragment (arrives token by token)
            if tc.function.arguments:
                tool_calls[tc.index]["arguments"] += tc.function.arguments

# Print the fully assembled tool call(s) once streaming is complete
print("\n\n--- Tool Call ---")
if tool_calls:
    for tc in tool_calls:
        print(f"Tool:      {tc['name']}")
        print(f"Arguments: {tc['arguments']}")
else:
    # Model chose to answer directly without calling a tool
    print("No tool call — model answered directly.")
