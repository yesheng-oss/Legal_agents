from dataclasses import dataclass
from pathlib import Path

import requests

from config import CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, TOP_K
from rag import LegalRAG


LEGAL_KEYWORDS = (
    "法律",
    "法条",
    "案例",
    "类案",
    "判",
    "刑",
    "罪",
    "责任",
    "合同",
    "赔偿",
    "起诉",
    "法院",
    "盗窃",
    "诈骗",
    "故意伤害",
)


@dataclass
class QuestionAnalysis:
    intent: str
    top_k: int
    reason: str


class LegalAgent:
    def __init__(self, rag=None):
        self._rag = rag

    @property
    def rag(self):
        if self._rag is None:
            self._rag = LegalRAG()
        return self._rag

    def analyze_question(self, question):
        text = question.strip()
        if not text or not any(keyword in text for keyword in LEGAL_KEYWORDS):
            return QuestionAnalysis("out_of_scope", 0, "问题不属于法律知识库问答范围")

        if any(keyword in text for keyword in ("类案", "类似", "案例", "案情")):
            return QuestionAnalysis("similar_case", 5, "用户需要查找相似案例")

        if any(keyword in text for keyword in ("判多久", "刑期", "量刑", "怎么判")):
            return QuestionAnalysis("sentencing_reference", 5, "用户需要量刑或判罚参考")

        return QuestionAnalysis("legal_qa", TOP_K, "用户需要法律问题分析")

    def retrieve(self, question, top_k=None):
        docs = self.rag.retrieve(question, k=top_k or TOP_K)
        return [self._format_reference(index + 1, doc) for index, doc in enumerate(docs)]

    def chat(self, question, history=None, memory=None):
        analysis = self.analyze_question(question)
        steps = ["问题分析"]

        if analysis.intent == "out_of_scope":
            return {
                "question": question,
                "intent": analysis.intent,
                "answer": "该问题超出中文法律知识库的回答范围，请输入法律咨询、类案检索或量刑参考相关问题。",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": steps,
            }

        try:
            docs = self.rag.retrieve(question, k=analysis.top_k)
        except FileNotFoundError:
            return {
                "question": question,
                "intent": analysis.intent,
                "answer": "向量库尚未构建，无法根据知识库可靠回答。请先运行 python src/download.py 和 python src/ingest.py 构建法律案例索引。",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": steps + ["案例检索"],
            }
        steps.append("案例检索")

        if not docs:
            return {
                "question": question,
                "intent": analysis.intent,
                "answer": "无法根据知识库可靠回答：当前没有检索到足够相关的法律案例依据。",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": steps + ["引用校验"],
            }

        try:
            answer = self.rag.generate_answer(question, docs, history=history or [], memory=memory or {})
        except TypeError:
            answer = self.rag.generate_answer(question, docs)
        steps.extend(["答案生成", "引用校验"])
        references = [self._format_reference(index + 1, doc) for index, doc in enumerate(docs)]

        checked_answer, confidence = self._check_citation(answer, references)
        return {
            "question": question,
            "intent": analysis.intent,
            "answer": checked_answer,
            "confidence": confidence,
            "risk_notice": self._risk_notice(),
            "references": references,
            "steps": steps,
        }

    def health(self):
        vector_store = "ok" if Path(CHROMA_PERSIST_DIR).exists() else "missing"
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama = "ok" if response.ok else "error"
        except requests.RequestException:
            ollama = "unavailable"

        status = "ok" if vector_store == "ok" and ollama == "ok" else "degraded"
        return {"status": status, "vector_store": vector_store, "ollama": ollama}

    def _check_citation(self, answer, references):
        if not references:
            return "无法根据知识库可靠回答：当前没有可引用的参考案例。", "low"

        if "[1]" not in answer and "参考案例" not in answer:
            answer = f"{answer}\n\n参考依据：见参考案例[1]。"

        return answer, "high"

    def _format_reference(self, ref_id, doc):
        metadata = getattr(doc, "metadata", {}) or {}
        return {
            "id": ref_id,
            "content": getattr(doc, "page_content", ""),
            "accusations": metadata.get("accusations", ""),
            "articles": metadata.get("articles", ""),
            "punishment": metadata.get("punishment", 0),
        }

    def _risk_notice(self):
        return "回答仅供学习参考，不构成正式法律意见；具体案件请咨询执业律师。"
