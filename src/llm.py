import json
import logging
import time
from typing import Generator, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from settings import get_settings

logger = logging.getLogger("legal_agent.llm")


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """LLM 调用基础异常。"""


class LLMConfigError(LLMError):
    """配置错误：缺少 API Key、不支持的 provider 等。"""


class LLMTimeoutError(LLMError):
    """LLM 请求超时。"""


class LLMRateLimitError(LLMError):
    """触发速率限制（429）。"""


class LLMResponseError(LLMError):
    """LLM 返回非 2xx 状态码或非法响应体。"""


# ---------------------------------------------------------------------------
# DeepSeek Provider
# ---------------------------------------------------------------------------

class DeepSeekProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        if not api_key:
            raise LLMConfigError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        settings = get_settings()
        self.api_key = api_key
        self.model = model or settings.deepseek_model
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _build_messages(prompt: str, history: Optional[list] = None) -> list:
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, history: Optional[list] = None) -> str:
        """阻塞式生成，内部走流式收集。兼容旧接口。"""
        return "".join(self.generate_stream(prompt, history))

    def generate_stream(
        self, prompt: str, history: Optional[list] = None
    ) -> Generator[str, None, None]:
        """SSE 流式生成 DeepSeek 回答，逐字 yield。"""
        messages = self._build_messages(prompt, history)
        logger.info(
            "[DeepSeek] request model=%s messages=%d timeout=%ds",
            self.model,
            len(messages),
            self.timeout,
        )
        t0 = time.time()
        try:
            resp = self._post_chat(messages)
        except requests.Timeout:
            logger.error("[DeepSeek] timeout after %.1fs", time.time() - t0)
            raise LLMTimeoutError(f"DeepSeek request timed out after {self.timeout}s")
        except requests.HTTPError as exc:
            self._handle_http_error(exc)
            raise  # _handle_http_error 内已抛异常，此行仅作兜底

        for chunk in self._parse_sse(resp):
            yield chunk

        logger.info("[DeepSeek] completed in %.2fs", time.time() - t0)

    @retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, LLMRateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _post_chat(self, messages: list):
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "stream": True,
            },
            timeout=self.timeout,
            stream=True,
        )
        if resp.status_code == 429:
            logger.warning("[DeepSeek] rate limited (429), will retry")
            raise LLMRateLimitError("DeepSeek rate limit exceeded (429)")
        resp.raise_for_status()
        return resp

    @staticmethod
    def _parse_sse(resp) -> Generator[str, None, None]:
        """解析 OpenAI / DeepSeek SSE 流。"""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    @staticmethod
    def _handle_http_error(exc: requests.HTTPError):
        status = exc.response.status_code if exc.response else 0
        text = exc.response.text[:500] if exc.response else ""
        logger.error("[DeepSeek] HTTP %d: %s", status, text)
        if status == 429:
            raise LLMRateLimitError(f"DeepSeek rate limit (429): {text}")
        if status >= 500:
            raise LLMResponseError(f"DeepSeek server error ({status}): {text}")
        raise LLMResponseError(f"DeepSeek request failed ({status}): {text}")


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

class OllamaProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ):
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, history: Optional[list] = None) -> str:
        """阻塞式生成，内部走流式收集。兼容旧接口。"""
        return "".join(self.generate_stream(prompt, history))

    def generate_stream(
        self, prompt: str, history: Optional[list] = None
    ) -> Generator[str, None, None]:
        """使用 /api/chat 端点流式生成，正确传递 messages 数组。"""
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        logger.info(
            "[Ollama] request model=%s messages=%d timeout=%ds",
            self.model,
            len(messages),
            self.timeout,
        )
        t0 = time.time()
        try:
            resp = self._post_chat(messages)
        except requests.Timeout:
            logger.error("[Ollama] timeout after %.1fs", time.time() - t0)
            raise LLMTimeoutError(f"Ollama request timed out after {self.timeout}s")
        except requests.HTTPError as exc:
            self._handle_http_error(exc)
            raise

        for chunk in self._parse_ndjson(resp):
            yield chunk

        logger.info("[Ollama] completed in %.2fs", time.time() - t0)

    @retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _post_chat(self, messages: list):
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"num_predict": self.max_tokens},
            },
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        return resp

    @staticmethod
    def _parse_ndjson(resp) -> Generator[str, None, None]:
        """解析 Ollama /api/chat 的 NDJSON 流。"""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            try:
                data = json.loads(line)
                if data.get("done"):
                    break
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
            except json.JSONDecodeError:
                continue

    @staticmethod
    def _handle_http_error(exc: requests.HTTPError):
        status = exc.response.status_code if exc.response else 0
        text = exc.response.text[:500] if exc.response else ""
        logger.error("[Ollama] HTTP %d: %s", status, text)
        if status >= 500:
            raise LLMResponseError(f"Ollama server error ({status}): {text}")
        raise LLMResponseError(f"Ollama request failed ({status}): {text}")


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_llm_provider(fallback_to_ollama: bool = True):
    settings = get_settings()
    if settings.llm_provider == "deepseek":
        try:
            return DeepSeekProvider(
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model,
                base_url=settings.deepseek_base_url,
            )
        except LLMConfigError:
            if fallback_to_ollama:
                logger.warning(
                    "[LLM] DeepSeek config missing, falling back to Ollama"
                )
                return OllamaProvider(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                )
            raise

    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    raise LLMConfigError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


# ---------------------------------------------------------------------------
# LangChain ChatModel 适配（Agent 专用）
# ---------------------------------------------------------------------------

def create_langchain_chat_model(temperature: float = 0.2):
    """返回支持 tool-calling 的 LangChain ChatModel，用于 Agent。

    Ollama 使用 langchain-ollama 的 ChatOllama；
    DeepSeek 使用 langchain-openai 的 ChatOpenAI（DeepSeek 为 OpenAI 兼容 API）。
    """
    settings = get_settings()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            timeout=settings.ollama_timeout,
            num_predict=settings.ollama_max_tokens,
        )

    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise LLMConfigError(
                "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
            timeout=settings.deepseek_timeout,
        )

    raise LLMConfigError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
