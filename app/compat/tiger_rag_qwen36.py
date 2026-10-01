from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI


class Qwen36TigerRagProvider:
    """Minimal invoke/stream contract used by Tiger-adv-Agentic-RAG."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:18081/v1",
        model: str = "qwen36",
        max_tokens: int = 256,
        temperature: float = 0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=900.0)

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("qwen36 returned an empty provider response.")
        return content.strip()

    def stream(self, prompt: str) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in response:
            for choice in chunk.choices:
                content = choice.delta.content
                if content:
                    yield content
