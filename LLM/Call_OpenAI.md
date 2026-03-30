
## Call OpenAI API

```from openai import OpenAI
client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a customer service assistant."},
        {"role": "user", "content": "Where is my order?"}
    ]
)
print(response.choices[0].message.content)
```


## Conversation Memory (pass history into messages)

```
def chat_with_memory():
    """Multi-turn conversation that retains history"""
    messages = [
        {"role": "system", "content": "You are a friendly assistant."}
    ]

    print("=== Multi-turn Conversation (type 'quit' to exit) ===")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"AI: {assistant_reply}\n")

    return messages
```