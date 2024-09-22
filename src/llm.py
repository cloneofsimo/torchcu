import os

from dotenv import load_dotenv
from together import Together

load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


class Input:
    def __init__(self):
        self.messages = []

    def add(self, content: str, role: str = "user"):
        self.messages.append({"role": role, "content": content})


class LLM:
    def __init__(self):
        self.usage = 0

    def generate(self, input: Input, **kwargs):
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=input.messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

            if chunk.usage is not None:
                self.usage += chunk.usage.total_tokens
