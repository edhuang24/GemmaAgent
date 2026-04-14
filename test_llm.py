from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

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

print(response.choices[0].message.content)