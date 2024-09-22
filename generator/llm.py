import logging
import os

from dotenv import load_dotenv
from together import Together

load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

logger = logging.getLogger()


class LLM:
    def __init__(self):
        self.usage = 0

    def generate(self, messages: list[dict[str, str]], **kwargs):
        logger.debug(
            f'Generating completion for {"\n".join([m["role"] + ": " + m["content"] for m in messages])}'
        )
        args = dict(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True,
        )
        args.update(kwargs)
        stream = client.chat.completions.create(
            messages=messages,
            **args,
        )

        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

            if chunk.usage is not None:
                self.usage += chunk.usage.total_tokens
