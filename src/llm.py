import requests

from settings import get_settings


class LLMConfigError(RuntimeError):
    pass


class DeepSeekProvider:
    def __init__(self, api_key=None, model=None, base_url=None, timeout=600):
        if not api_key:
            raise LLMConfigError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        settings = get_settings()
        self.api_key = api_key
        self.model = model or settings.deepseek_model
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt, history=None):
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class OllamaProvider:
    def __init__(self, model=None, base_url=None, timeout=600, max_tokens=1024):
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, prompt, history=None):
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in (history or []))
        full_prompt = f"历史对话：\n{history_text}\n\n当前任务：\n{prompt}" if history_text else prompt
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_predict": self.max_tokens},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]


def create_llm_provider(fallback_to_ollama=True):
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
                return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)
            raise

    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)

    raise LLMConfigError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
