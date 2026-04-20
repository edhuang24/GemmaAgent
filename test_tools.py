# json is needed to parse tool call arguments, which arrive as a JSON string
import json

# Use the openai client pointed at LM Studio's local endpoint
from openai import OpenAI

# Import all tool functions so we can actually execute them
from tools import rag_retrieve, web_search, read_file, write_file, run_shell

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

# Maps tool name strings to the actual Python functions that implement them
TOOL_REGISTRY = {
    "rag_retrieve": rag_retrieve,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
}

# The query you want to test — swap this out to try different tools
query = "What's the weather in San Francisco like today?"

# Build the initial message history with just the user's query
messages = [{"role": "user", "content": query}]

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
    messages=messages,
)

print("--- Thinking ---")

# Tool call data arrives in fragments across chunks — accumulate it here
tool_calls = []

# We also need to capture any plain text content in case the model answers directly
response_content = ""

# Iterate over each chunk as it arrives from the model
for chunk in stream:
    delta = chunk.choices[0].delta

    # Gemma 4's thinking tokens come through in reasoning_content, not content
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)

    # Plain text content streams in via delta.content — print and accumulate
    if delta.content:
        print(delta.content, end="", flush=True)
        # Accumulate content in case the model answers directly without a tool call
        response_content += delta.content

    # Tool call data is split across chunks — each chunk adds a fragment
    if delta.tool_calls:
        for tc in delta.tool_calls:
            # tc.index tells us which tool call this fragment belongs to
            # If it's a new index, start a new entry in our accumulator
            if tc.index >= len(tool_calls):
                # Store the id too — the API requires it when we send back the tool result
                tool_calls.append({"id": tc.id, "name": "", "arguments": ""})
            # Append the name fragment (arrives first, usually in one chunk)
            if tc.function.name:
                tool_calls[tc.index]["name"] += tc.function.name
            # Append the arguments fragment (arrives token by token)
            if tc.function.arguments:
                tool_calls[tc.index]["arguments"] += tc.function.arguments

print("\n\n--- Tool Call ---")

if tool_calls:
    for tc in tool_calls:
        print(f"Tool:      {tc['name']}")
        print(f"Arguments: {tc['arguments']}")

    # Append Gemma's tool call response to history so the next call has full context
    # The API expects the assistant message to include the tool_calls list
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ],
    })

    # Execute each tool and append its result to the message history
    for tc in tool_calls:
        # Parse the JSON argument string into a dict so we can call the function
        tool_args = json.loads(tc["arguments"])
        # Look up the function in the registry and call it with the parsed args
        tool_result = TOOL_REGISTRY[tc["name"]](**tool_args)

        print(f"\n--- Tool Result (preview) ---\n{tool_result[:300]}")

        # Append the tool result with role "tool" — the API requires this format
        # tool_call_id links this result back to the specific tool call that triggered it
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_result,
        })

    # Send the full message history back to get Gemma's final answer
    final_response = client.chat.completions.create(
        model="gemma",
        tools=tools,
        max_tokens=512,
        messages=messages,
    )

    print(f"\n--- Final Answer ---\n{final_response.choices[0].message.content}")

else:
    # Model chose to answer directly without calling a tool
    print("No tool call — model answered directly.")
    print(f"\n--- Final Answer ---\n{response_content}")
