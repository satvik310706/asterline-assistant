"""Small provider abstraction with an offline deterministic FakeLLM."""

from __future__ import annotations

import json
import os
import urllib.request

from google import genai

from .models import AnswerDraft, EvidencePack


class LLMProvider:
    name = "unknown"
    model = "unknown"

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        raise NotImplementedError


class FakeLLM(LLMProvider):
    name = "fake"
    model = "deterministic"

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        self.calls += 1
        if not evidence.chunks:
            return AnswerDraft(answer="I could not find enough evidence in the supplied corpus.", abstained=True)
        citations = [f"[{item.chunk.chunk_id} — {item.chunk.title}]" for item in evidence.chunks[:2]]
        text = " ".join(item.chunk.text for item in evidence.chunks[:2])
        return AnswerDraft(answer=f"Based on the supplied corpus: {text} {' '.join(citations)}", cited_chunk_ids=[item.chunk.chunk_id for item in evidence.chunks[:2]])


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        self.calls += 1
        context = "\n".join(f"{x.chunk.chunk_id} | {x.chunk.title} | {x.chunk.text}" for x in evidence.chunks)
        prompt = f"Answer only from this evidence. Cite each claim as [chunk_id — title]. If insufficient, abstain.\nQuestion: {question}\nEvidence:\n{context}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0, "response_mime_type": "application/json", "response_schema": AnswerDraft},
        )
        return AnswerDraft.model_validate(json.loads(response.text or "{}"))


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        self.calls += 1
        context = "\n".join(f"{x.chunk.chunk_id}: {x.chunk.text}" for x in evidence.chunks)
        payload = json.dumps({"model": self.model, "prompt": f"Answer briefly from evidence only. Question: {question}\n{context}", "stream": False, "format": "json"}).encode()
        request = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
        return AnswerDraft.model_validate(json.loads(result["response"]))


def make_provider(provider: str, gemini_model: str, ollama_model: str) -> LLMProvider:
    if provider == "fake" or (provider == "auto" and not os.getenv("GEMINI_API_KEY")):
        return FakeLLM()
    if provider in {"auto", "gemini"}:
        return GeminiLLM(gemini_model)
    return OllamaLLM(ollama_model)

