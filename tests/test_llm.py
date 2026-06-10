import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm import DeepSeekProvider, LLMConfigError, OllamaProvider, create_llm_provider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.ok = True

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_deepseek_provider_uses_openai_compatible_chat_completions(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse({"choices": [{"message": {"content": "deepseek answer"}}]})

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-v4-flash", base_url="https://api.deepseek.com")

    answer = provider.generate("hello", history=[{"role": "user", "content": "previous"}])

    assert answer == "deepseek answer"
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"]["model"] == "deepseek-v4-flash"
    assert calls[0]["json"]["messages"][0]["role"] == "user"


def test_deepseek_provider_requires_api_key():
    with pytest.raises(LLMConfigError):
        DeepSeekProvider(api_key="")


def test_ollama_provider_keeps_local_generate_api(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"response": "ollama answer"})

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = OllamaProvider(model="qwen2.5:7b", base_url="http://localhost:11434")

    answer = provider.generate("hello")

    assert answer == "ollama answer"
    assert calls[0]["url"] == "http://localhost:11434/api/generate"


def test_create_provider_can_fallback_to_ollama_when_deepseek_key_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    provider = create_llm_provider(fallback_to_ollama=True)

    assert isinstance(provider, OllamaProvider)
