from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K
from llm import create_llm_provider


PROMPT = """你是一名中国法律知识检索增强问答助手。
请只基于“法律案例参考、历史对话、案件记忆”回答用户问题；如果依据不足，请明确说明无法可靠判断，并列出需要补充的证据。

历史对话：
{history}

案件记忆：
{memory}

法律案例参考：
{context}

用户问题：
{question}

请输出结构化中文法律分析，包含：核心结论、依据说明、参考案例、风险提示。"""


class LegalRAG:
    def __init__(self, llm_provider=None):
        persist = Path(CHROMA_PERSIST_DIR)
        if not persist.exists():
            raise FileNotFoundError(f"Chroma index not found at {CHROMA_PERSIST_DIR}. Run ingest.py first.")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = Chroma(
            collection_name="legal_docs",
            persist_directory=str(persist),
            embedding_function=embeddings,
        )
        self.llm_provider = llm_provider or create_llm_provider()

    def retrieve(self, question, k=None):
        return self.db.similarity_search(question, k=k or TOP_K)

    def generate_answer(self, question, docs, history=None, memory=None):
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        history_text = self._format_history(history or [])
        memory_text = self._format_memory(memory or {})
        prompt = PROMPT.format(context=context, question=question, history=history_text, memory=memory_text)
        return self.llm_provider.generate(prompt)

    def query(self, question):
        docs = self.retrieve(question)
        answer = self.generate_answer(question, docs)
        return answer, docs

    def _format_history(self, history):
        if not history:
            return "无"
        labels = {"user": "用户", "assistant": "助手", "system": "系统"}
        return "\n".join(
            f"{labels.get(item.get('role', 'unknown'), item.get('role', 'unknown'))}：{self._trim_prompt_text(item.get('content', ''))}"
            for item in history
        )

    def _format_memory(self, memory):
        if not memory:
            return "无"
        labels = {
            "facts_summary": "事实摘要",
            "user_goal": "用户目标",
            "dispute_focus": "争议焦点",
            "confirmed_points": "已确认结论",
            "missing_evidence": "待补充证据",
        }
        return "\n".join(f"{labels.get(key, key)}：{self._trim_prompt_text(str(value))}" for key, value in memory.items() if value)

    def _trim_prompt_text(self, text, max_chars=360):
        text = text or ""
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."


if __name__ == "__main__":
    rag = LegalRAG()
    print("Legal RAG ready. Type 'quit' to exit.")
    while True:
        q = input("\n问题: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer, refs = rag.query(q)
        print(f"\n答案: {answer}")
        print(f"\n参考案例: {len(refs)} 条")
