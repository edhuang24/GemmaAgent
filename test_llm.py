# Use the openai SDK — LM Studio exposes an OpenAI-compatible API, so no custom client needed
from openai import OpenAI
import json

# Point the client at LM Studio's local server instead of OpenAI's servers.
# The api_key value doesn't matter here — LM Studio requires the field but doesn't validate it.
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# Send a minimal chat completion request to verify the local model is reachable and responding.
# temperature=0.0 makes the output deterministic — useful for testing since the answer won't vary.
response = client.chat.completions.create(
    model="gemma-4-26b-a4b",
    messages=[{
        "role": "user",
        "content": "What is 2 + 2? Answer in one sentence."
    }],
    temperature=0.0,
)

# breakpoint()

# Uncomment below to easily see response object (type ChatCompletion) in readable JSON
# print(response.model_dump_json(indent=4))

# The response contains a list of choices — we take the first (and only) one and print the model's reply
print(response.choices[0].message.content)