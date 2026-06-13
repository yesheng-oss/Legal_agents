import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm import (
    DeepSeekProvider,
    LLMConfigError,
    LLMResponseError,
    LLMTimeoutError,
    OllamaProvider,
    create_llm_provider,
)


class FakeResponse:
    """模拟 requests 的流式响应。"""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, str):
                yield line.encode("utf-8")
            else:
                yield line


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

def test_deepseek_provider_uses_openai_compatible_chat_completions(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout, "stream": stream})
        # SSE 流式格式
        lines = [
            'data: {"choices":[{"delta":{"content":"deepseek"}}]}',
            'data: {"choices":[{"delta":{"content":" answer"}}]}',
            "data: [DONE]",
        ]
        return FakeResponse(lines)

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = DeepSeekProvider(
        api_key="sk-test", model="deepseek-v4-flash", base_url="https://api.deepseek.com"
    )

    answer = provider.generate("hello", history=[{"role": "user", "content": "previous"}])

    assert answer == "deepseek answer"
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"]["model"] == "deepseek-v4-flash"
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["messages"][0]["role"] == "user"


def test_deepseek_provider_supports_streaming(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls.append({"url": url})
        lines = [
            'data: {"choices":[{"delta":{"content":"chunk1"}}]}',
            'data: {"choices":[{"delta":{"content":"chunk2"}}]}',
            "data: [DONE]",
        ]
        return FakeResponse(lines)

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-v4-flash")

    chunks = list(provider.generate_stream("test"))

    assert chunks == ["chunk1", "chunk2"]


def test_deepseek_provider_raises_on_timeout(monkeypatch):
    import requests

    def fake_post(*args, **kwargs):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-v4-flash")

    with pytest.raises(LLMTimeoutError):
        provider.generate("test")


def test_deepseek_provider_raises_on_http_error(monkeypatch):
    class ErrResponse:
        status_code = 500
        text = "internal error"

        def raise_for_status(self):
            import requests

            raise requests.HTTPError("500", response=self)

    def fake_post(*args, **kwargs):
        return ErrResponse()

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-v4-flash")

    with pytest.raises(LLMResponseError):
        provider.generate("test")


def test_deepseek_provider_requires_api_key():
    with pytest.raises(LLMConfigError):
        DeepSeekProvider(api_key="")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def test_ollama_provider_uses_chat_endpoint_with_messages(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, "json": kwargs.get("json"), "timeout": kwargs.get("timeout"), "stream": kwargs.get("stream")})
        # Ollama /api/chat NDJSON 格式
        import json as _json
        lines = [
            _json.dumps({"message": {"role": "assistant", "content": "ollama"}, "done": False}),
            _json.dumps({"message": {"role": "assistant", "content": " answer"}, "done": False}),
            _json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        return FakeResponse(lines)

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = OllamaProvider(model="qwen2.5:7b", base_url="http://localhost:11434")

    answer = provider.generate("hello", history=[{"role": "user", "content": "prev"}])

    assert answer == "ollama answer"
    assert calls[0]["url"] == "http://localhost:11434/api/chat"
    assert calls[0]["json"]["model"] == "qwen2.5:7b"
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["messages"][-1]["role"] == "user"
    assert calls[0]["json"]["messages"][-1]["content"] == "hello"


def test_ollama_provider_supports_streaming(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url})
        import json as _json
        lines = [
            _json.dumps({"message": {"role": "assistant", "content": "a"}, "done": False}),
            _json.dumps({"message": {"role": "assistant", "content": "b"}, "done": False}),
            _json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        return FakeResponse(lines)

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = OllamaProvider(model="qwen2.5:7b")

    chunks = list(provider.generate_stream("test"))

    assert chunks == ["a", "b"]


def test_ollama_provider_raises_on_timeout(monkeypatch):
    import requests

    def fake_post(*args, **kwargs):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr("llm.requests.post", fake_post)
    provider = OllamaProvider(model="qwen2.5:7b")

    with pytest.raises(LLMTimeoutError):
        provider.generate("test")


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def test_create_provider_can_fallback_to_ollama_when_deepseek_key_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    provider = create_llm_provider(fallback_to_ollama=True)

    assert isinstance(provider, OllamaProvider)
