from openai import OpenAI

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

print(response.choices[0].message.content)