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
query = "What happened with the Iran War today?"

# Cap how many tool calls we execute in this test — prevents runaway multi-call responses
MAX_TOOL_CALLS = 5

# System prompt instructs Gemma to call only one tool at a time — prevents the multi-call runaway we saw earlier
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Use tools when you need to look something up, read a file, search the web, or run a command.
Only call one tool at a time.
Once you have enough information to answer the user's question, stop calling tools and give a final answer.
Do not call the same tool with the same arguments twice."""

# Build the initial message history with the system prompt and the user's query
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query},
]

# DEBUG: print the full payload being sent to the LLM, then pause before firing the request
import json
print("\n--- DEBUG: Outgoing Payload ---")
print(json.dumps(messages, indent=2))
input("Press Enter to send to LLM...")

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

# DEBUG: print the full message history after the stream, including Gemma's assembled response
# print("\n--- DEBUG: Message History After Stream ---")
# print(json.dumps(messages, indent=2))
# input("Press Enter to continue...")

print("\n\n--- Tool Call ---")

if tool_calls:
    # Limit to MAX_TOOL_CALLS — Gemma often generates more than needed despite the system prompt
    capped_tool_calls = tool_calls[:MAX_TOOL_CALLS]
    if len(tool_calls) > MAX_TOOL_CALLS:
        print(f"(Capping to {MAX_TOOL_CALLS} tool call(s) — model generated {len(tool_calls)} total)")

    for tc in capped_tool_calls:
        print(f"Tool:      {tc['name']}")
        print(f"Arguments: {tc['arguments']}")

    # Append Gemma's tool call response to history so the next call has full context
    # The API expects the assistant message to include the full list of tool calls it generated
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in capped_tool_calls
        ],
    })

    # Execute each capped tool call and append its result to the message history
    for tc in capped_tool_calls:
        # Guard against truncated tool calls — max_tokens can cut off arguments mid-generation
        if not tc["arguments"].strip():
            print(f"\nSkipping {tc['name']} — arguments were truncated (hit max_tokens limit)")
            continue
        # Parse the JSON argument string into a dict so we can call the function
        tool_args = json.loads(tc["arguments"])
        # Look up the function in the registry and call it with the parsed args
        tool_result = TOOL_REGISTRY[tc["name"]](**tool_args)

        print(f"\n--- Tool Result (preview) ---\n{tool_result[:1000]}")

        # Append the tool result with role "tool" — the API requires this format
        # tool_call_id links this result back to the specific tool call that triggered it
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_result,
        })

    # Send the full message history back and stream the final answer
    # Higher max_tokens gives the model enough room to summarize the tool results
    final_stream = client.chat.completions.create(
        model="gemma",
        tools=tools,
        max_tokens=2560,
        stream=True,
        messages=messages,
    )

    print("\n--- Reasoning ---")

    # Accumulate tool calls and content from the final stream
    final_tool_calls = []
    final_content = ""

    # Iterate over the final stream, printing reasoning and content as they arrive
    for chunk in final_stream:
        delta = chunk.choices[0].delta

        # Print reasoning tokens in real time as they stream in
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            print(delta.reasoning_content, end="", flush=True)

        # Accumulate plain text content for the final answer
        if delta.content:
            final_content += delta.content

        # Accumulate any tool calls in case the model wants another tool
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.index >= len(final_tool_calls):
                    final_tool_calls.append({"name": "", "arguments": ""})
                if tc.function.name:
                    final_tool_calls[tc.index]["name"] += tc.function.name
                if tc.function.arguments:
                    final_tool_calls[tc.index]["arguments"] += tc.function.arguments

    # Check if the model wants to call another tool instead of giving a final answer
    # This is normal ReAct behavior — the full agent loop in agent.py will handle this properly
    if final_tool_calls:
        print("\n\n--- Model wants to call another tool (needs more iterations) ---")
        for tc in final_tool_calls:
            print(f"Tool:      {tc['name']}")
            print(f"Arguments: {tc['arguments']}")
    else:
        print(f"\n\n--- Final Answer ---\n{final_content}")

else:
    # Model chose to answer directly without calling a tool
    print("No tool call — model answered directly.")
    print(f"\n--- Final Answer ---\n{response_content}")
