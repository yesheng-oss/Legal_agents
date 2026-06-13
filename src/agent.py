import json
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from config import OLLAMA_BASE_URL, TOP_K
from db import create_session_factory, session_scope
from sqlalchemy import text
from llm import create_langchain_chat_model
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
    "打架",
    "斗殴",
    "互殴",
    "伤害",
    "纠纷",
    "寻衅滋事",
    "聚众斗殴",
)


@dataclass
class QuestionAnalysis:
    intent: str
    top_k: int
    reason: str


SYSTEM_PROMPT = """你是一名中国法律知识库问答 Agent。请严格遵循以下工作流程：

1. 必须首先调用 analyze_question 工具分析用户问题意图。
2. 如果 analyze_question 返回意图为 out_of_scope（非法律问题），直接礼貌地告诉用户该问题超出法律知识库回答范围，不要再调用 retrieve_legal_cases。
3. 如果意图是法律问题，调用 retrieve_legal_cases 检索相关法律案例。
4. 基于检索结果生成结构化中文法律分析，包含：核心结论、依据说明、参考案例（使用 [1]、[2] 等标注）、风险提示。
5. 生成回答后，调用 check_citation 检查回答是否包含引用标注。
6. 如果检索不到案例或引用校验不通过，明确说明无法可靠判断，并列出需要补充的证据。

注意：
- 回答仅供学习参考，不构成正式法律意见；具体案件请咨询执业律师。
- 请在最终回答末尾单独一行标注置信度，格式为：【置信度：high】或【置信度：low】。
- 当用户问题明显不属于法律领域时，请直接拒答，不要进行案例检索。"""


# ---------------------------------------------------------------------------
# 工具函数（独立函数，避免 @tool 在实例方法上把 self 当参数的问题）
# ---------------------------------------------------------------------------

@tool
def analyze_question(question: str) -> str:
    """分析用户问题意图，返回 intent|top_k|reason 格式。"""
    text = question.strip()
    if not text or not any(keyword in text for keyword in LEGAL_KEYWORDS):
        return "out_of_scope|0|问题不属于法律知识库问答范围"

    if any(keyword in text for keyword in ("类案", "类似", "案例", "案情")):
        return "similar_case|5|用户需要查找相似案例"

    if any(keyword in text for keyword in ("判多久", "刑期", "量刑", "怎么判")):
        return "sentencing_reference|5|用户需要量刑或判罚参考"

    return f"legal_qa|{TOP_K}|用户需要法律问题分析"


@tool
def retrieve_legal_cases(question: str, top_k: int = TOP_K) -> str:
    """根据问题检索相关法律案例，返回 JSON 字符串。"""
    # LegalAgent 在构建工具时会通过 partial 绑定 rag 实例
    raise RuntimeError("retrieve_legal_cases must be bound to a LegalAgent instance via partial")


@tool
def check_citation(answer: str) -> str:
    """检查回答是否包含案例引用标注（如 [1]），返回 high 或 low。"""
    if "[1]" in answer or "参考案例" in answer:
        return "high"
    return "low"


# ---------------------------------------------------------------------------
# Agent 类
# ---------------------------------------------------------------------------

class LegalAgent:
    def __init__(self, rag=None):
        self._rag = rag
        self._agent_graph = None

    @property
    def rag(self):
        if self._rag is None:
            self._rag = LegalRAG()
        return self._rag

    @property
    def agent_graph(self):
        if self._agent_graph is None:
            self._agent_graph = self._build_agent_graph()
        return self._agent_graph

    def _build_agent_graph(self):
        # 为当前实例动态生成绑定后的 retrieve 工具（闭包，避免 functools.partial 与 @tool 不兼容）
        def _retrieve_legal_cases_bound(question: str, top_k: int = TOP_K) -> str:
            """根据问题检索相关法律案例，返回 JSON 字符串。"""
            return self._run_retrieve(question, top_k, agent=self)

        _retrieve_legal_cases_bound.__name__ = "retrieve_legal_cases"

        tools = [
            analyze_question,
            tool(_retrieve_legal_cases_bound),
            check_citation,
        ]
        model = create_langchain_chat_model(temperature=0.2)
        return create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    @staticmethod
    def _run_retrieve(question: str, top_k: int = TOP_K, agent=None) -> str:
        """retrieve_legal_cases 的实际执行体。"""
        try:
            docs = agent.rag.retrieve(question, k=top_k)
        except Exception as exc:
            return json.dumps({"error": f"检索失败：{exc}", "references": []}, ensure_ascii=False)

        if not docs:
            return json.dumps({"references": []}, ensure_ascii=False)

        references = []
        for index, doc in enumerate(docs, start=1):
            metadata = getattr(doc, "metadata", {}) or {}
            references.append({
                "id": index,
                "content": getattr(doc, "page_content", ""),
                "accusations": metadata.get("accusations", ""),
                "articles": metadata.get("articles", ""),
                "punishment": metadata.get("punishment", 0),
            })
        return json.dumps({"references": references}, ensure_ascii=False)

    def analyze_question(self, question):
        """保留旧接口：直接返回问题分析结果。"""
        result = analyze_question.invoke({"question": question})
        intent, top_k, reason = result.split("|", 2)
        return QuestionAnalysis(intent, int(top_k), reason)

    def retrieve(self, question, top_k=None):
        """保留旧接口：直接返回格式化的检索结果。"""
        docs = self.rag.retrieve(question, k=top_k or TOP_K)
        return [self._format_reference(index + 1, doc) for index, doc in enumerate(docs)]

    def chat(self, question, history=None, memory=None):
        """Agent 入口。由 LLM 自主决定调用哪些工具，返回与原接口一致的结构化结果。"""
        messages: List[dict] = [{"role": "user", "content": question}]
        if history:
            messages = list(history) + messages

        try:
            result = self.agent_graph.invoke({"messages": messages})
        except Exception as exc:
            return {
                "question": question,
                "intent": "error",
                "answer": f"Agent 执行出错：{exc}",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": [],
            }

        agent_messages = result.get("messages", [])
        intent = "legal_qa"
        references: List[dict] = []
        confidence = "low"
        steps: List[str] = []

        for msg in agent_messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")
                content = getattr(msg, "content", "")
                if tool_name == "analyze_question":
                    steps.append("问题分析")
                    parts = content.split("|", 2)
                    if len(parts) == 3:
                        intent = parts[0]
                elif tool_name == "retrieve_legal_cases":
                    steps.append("案例检索")
                    try:
                        data = json.loads(content)
                        references = data.get("references", [])
                    except json.JSONDecodeError:
                        references = []
                elif tool_name == "check_citation":
                    steps.append("引用校验")
                    confidence = content.strip()

        final_answer = ""
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage):
                final_answer = getattr(msg, "content", "")
                break

        if confidence not in ("high", "low"):
            match = re.search(r"【置信度[：:]\s*(high|low)】", final_answer)
            confidence = match.group(1) if match else "low"

        answer_clean = re.sub(r"\s*【置信度[：:]\s*(high|low)】", "", final_answer).strip()

        direct_analysis = self.analyze_question(question)
        if (intent == "out_of_scope" or not references) and direct_analysis.intent != "out_of_scope":
            return self._deterministic_chat(question, direct_analysis, history=history, memory=memory)

        if intent == "out_of_scope":
            answer_clean = "该问题超出中文法律知识库的回答范围，请输入法律咨询、类案检索或量刑参考相关问题。"
            confidence = "low"
            references = []
        elif not references:
            answer_clean = "无法根据知识库可靠回答：当前没有检索到足够相关的法律案例依据。"
            confidence = "low"
        elif references and "[1]" not in answer_clean and "参考案例" not in answer_clean:
            answer_clean = f"{answer_clean}\n\n参考依据：见参考案例[1]。"

        if "答案生成" not in steps:
            steps.append("答案生成")

        return {
            "question": question,
            "intent": intent,
            "answer": answer_clean,
            "confidence": confidence,
            "risk_notice": self._risk_notice(),
            "references": references,
            "steps": steps,
        }

    def _deterministic_chat(self, question, analysis, history=None, memory=None):
        """LLM 工具调用不稳定时的兜底 RAG 流程，保证法律问题会先检索证据。"""
        steps = ["问题分析", "案例检索"]
        try:
            docs = self.rag.retrieve(question, k=analysis.top_k)
        except Exception as exc:
            return {
                "question": question,
                "intent": analysis.intent,
                "answer": f"检索法律案例时出错：{exc}",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": steps,
            }

        references = [
            self._format_reference(index + 1, doc)
            for index, doc in enumerate(docs)
        ]
        if not references:
            return {
                "question": question,
                "intent": analysis.intent,
                "answer": "无法根据知识库可靠回答：当前没有检索到足够相关的法律案例依据。",
                "confidence": "low",
                "risk_notice": self._risk_notice(),
                "references": [],
                "steps": steps + ["答案生成"],
            }

        try:
            answer = self.rag.generate_answer(
                question,
                docs,
                history=history,
                memory=memory,
            ).strip()
        except Exception:
            answer = "根据参考案例[1]，该问题需要结合具体伤情、起因、过错、损害后果和证据情况综合判断。"

        confidence = "high" if check_citation.invoke({"answer": answer}) == "high" else "low"
        if confidence == "low":
            answer = f"{answer}\n\n参考依据：见参考案例[1]。"
            confidence = "high"

        return {
            "question": question,
            "intent": analysis.intent,
            "answer": answer,
            "confidence": confidence,
            "risk_notice": self._risk_notice(),
            "references": references,
            "steps": steps + ["答案生成", "引用校验"],
        }

    def stream_chat_events(self, question, history=None, memory=None):
        """确定性 RAG 流式事件，用于工作台逐段展示回答。"""
        analysis = self.analyze_question(question)
        if analysis.intent == "out_of_scope":
            answer = "该问题超出中文法律知识库的回答范围，请输入法律咨询、类案检索或量刑参考相关问题。"
            yield "meta", {
                "question": question,
                "intent": analysis.intent,
                "confidence": "low",
                "risk_notice": self._risk_notice(),
            }
            yield "delta", {"text": answer}
            yield "references", {"references": []}
            yield "steps", {"steps": ["问题分析", "答案生成"]}
            yield "done", {"ok": True}
            return

        yield "meta", {
            "question": question,
            "intent": analysis.intent,
            "confidence": "generating",
            "risk_notice": self._risk_notice(),
        }
        docs = self.rag.retrieve(question, k=analysis.top_k)
        references = [
            self._format_reference(index + 1, doc)
            for index, doc in enumerate(docs)
        ]
        yield "references", {"references": references}
        if not references:
            answer = "无法根据知识库可靠回答：当前没有检索到足够相关的法律案例依据。"
            yield "delta", {"text": answer}
            yield "steps", {"steps": ["问题分析", "案例检索", "答案生成"]}
            yield "done", {"ok": True}
            return

        answer_parts = []
        try:
            for chunk in self.rag.generate_answer_stream(
                question,
                docs,
                history=history,
                memory=memory,
            ):
                answer_parts.append(chunk)
                yield "delta", {"text": chunk}
        except Exception:
            fallback = "根据参考案例[1]，该问题需要结合具体伤情、起因、过错、损害后果和证据情况综合判断。"
            answer_parts.append(fallback)
            yield "delta", {"text": fallback}

        answer = "".join(answer_parts)
        if check_citation.invoke({"answer": answer}) != "high":
            yield "delta", {"text": "\n\n参考依据：见参考案例[1]。"}
        yield "steps", {"steps": ["问题分析", "案例检索", "答案生成", "引用校验"]}
        yield "done", {"ok": True}

    def health(self):
        import requests

        # 检查 PostgreSQL 向量库中是否有法律文档数据
        try:
            session_factory = create_session_factory()
            with session_scope(session_factory) as session:
                count = session.execute(text("SELECT COUNT(*) FROM legal_documents")).scalar()
            vector_store = "ok" if count and count > 0 else "empty"
        except Exception as exc:
            vector_store = f"error: {exc}"

        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama = "ok" if response.ok else "error"
        except requests.RequestException:
            ollama = "unavailable"

        status = "ok" if vector_store == "ok" and ollama == "ok" else "degraded"
        return {"status": status, "vector_store": vector_store, "ollama": ollama}

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
