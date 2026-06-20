# json is needed to parse tool call arguments, which arrive as a JSON string
import json

# OpenAI client pointed at LM Studio's local API
from openai import OpenAI

# Import all tool functions — these are what actually run when the agent calls a tool
from tools import rag_retrieve, web_search, read_file, write_file, run_shell

# Import all schemas — these are what the LLM sees to know what tools are available
from tools import (
    RAG_RETRIEVE_SCHEMA,
    WEB_SEARCH_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RUN_SHELL_SCHEMA,
)

# Point the client at LM Studio instead of the real OpenAI API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Bundle all schemas into a list to pass to the LLM on every call
TOOLS = [
    RAG_RETRIEVE_SCHEMA,
    WEB_SEARCH_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RUN_SHELL_SCHEMA,
]

# The system prompt shapes Gemma's behavior for the entire conversation
# It tells the model what it is, what tools it has, and when to stop looping
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Use tools when you need to look something up, read a file, search the web, or run a command.
Only call one tool at a time.
Once you have enough information to answer the user's question, stop calling tools and give a final answer.
Do not call the same tool with the same arguments twice."""

# Maps tool name strings to the actual Python functions that implement them
# When the LLM returns a tool call, we use this dict to find and run the right function
TOOL_REGISTRY = {
    "rag_retrieve": rag_retrieve,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
}


def run_agent(query: str, max_iterations: int = 10, max_tool_calls: int = 5) -> str:
    # Start the message history with the system prompt and the user's query
    # Every call to the LLM will include this full history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # Track how many tool calls have been executed across all iterations
    tool_call_count = 0

    # Loop up to max_iterations times — prevents infinite tool-calling cycles
    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        # Send the full message history and tool schemas to Gemma
        response = client.chat.completions.create(
            model="gemma",
            # tool_choice="auto" lets Gemma decide whether to call a tool or answer directly
            tool_choice="auto",
            tools=TOOLS,
            # Cap tokens per call to prevent runaway tool call generation
            max_tokens=2560,
            messages=messages,
        )

        # Extract the message object from the response
        message = response.choices[0].message

        # Print reasoning content if present so we can follow Gemma's thought process
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            print(f"Reasoning: {message.reasoning_content[:300]}...")

        # Append Gemma's response to history so future iterations have full context
        messages.append(message)

        # If Gemma returned a tool call, execute it and feed the result back
        if message.tool_calls:
            # Stop executing tools if we've hit the cap — force a final answer next iteration
            if tool_call_count >= max_tool_calls:
                print(f"Max tool calls ({max_tool_calls}) reached — stopping tool execution.")
                return "Max tool calls reached without a final answer."

            # Only use the first tool call — prevents runaway multi-call responses
            tool_call = message.tool_calls[0]

            # The tool name tells us which function to run
            tool_name = tool_call.function.name
            # Arguments arrive as a JSON string — parse them into a dict
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Tool call {tool_call_count + 1}/{max_tool_calls}: {tool_name}({tool_args})")

            # Look up the function in the registry and call it with the parsed args
            tool_fn = TOOL_REGISTRY[tool_name]
            tool_result = tool_fn(**tool_args)

            print(f"Tool result: {tool_result[:200]}...")

            # Append the tool result to history with role "tool"
            # The tool_call_id links this result back to the specific tool call that triggered it
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

            # Increment the counter after each successful tool execution
            tool_call_count += 1

        # If there are no tool calls, Gemma has produced a final answer — return it
        else:
            return message.content

    # If we exit the loop without a final answer, return what we have
    return "Max iterations reached without a final answer."


# Only run when this file is executed directly, not when imported by another module
if __name__ == "__main__":
    # Test query — swap this out to try different tools and scenarios
    result = run_agent("When was Rome founded and when was modern Italy founded?")
    print(f"\n--- Final Answer ---\n{result}")
