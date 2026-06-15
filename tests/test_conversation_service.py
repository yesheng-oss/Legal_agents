import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversation_service import ConversationService
from db import create_session_factory, init_db
from memory import MemoryService
from models import Base


class FakeAgent:
    def chat(self, question, history=None, memory=None):
        history = history or []
        memory = memory or {}
        return {
            "question": question,
            "intent": "legal_qa",
            "answer": f"回答：{question}；历史轮数：{len(history)}；记忆：{memory.get('facts_summary', '')}",
            "confidence": "high",
            "risk_notice": "回答仅供学习参考。",
            "references": [{"id": 1, "content": "案例", "accusations": "盗窃罪", "articles": "[264]", "punishment": 1}],
            "steps": ["问题分析", "案例检索", "答案生成", "引用校验"],
        }


class FakeMemoryExtractor:
    def extract(self, question, answer, current_memory):
        return {
            "facts_summary": f"用户咨询：{question}",
            "user_goal": "获得法律风险判断",
            "dispute_focus": "法律责任认定",
            "confirmed_points": answer[:80],
            "missing_evidence": "需补充证据材料。",
        }


def make_service():
    session_factory = create_session_factory("sqlite+pysqlite:///:memory:")
    init_db(Base.metadata, session_factory)
    return ConversationService(
        session_factory=session_factory,
        agent=FakeAgent(),
        memory_service=MemoryService(llm_extractor=FakeMemoryExtractor()),
    )


def test_chat_creates_case_conversation_messages_and_memory():
    service = make_service()

    result = service.chat(question="盗窃罪怎么判？")

    assert result["case_id"]
    assert result["conversation_id"]
    assert result["answer"].startswith("回答：盗窃罪")
    assert result["memory"]["facts_summary"]
    assert len(result["references"]) == 1

    conversation = service.get_conversation(result["conversation_id"])
    assert len(conversation["messages"]) == 2
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][1]["role"] == "assistant"


def test_chat_reuses_conversation_and_injects_recent_history():
    service = make_service()
    first = service.chat(question="盗窃罪怎么判？")

    second = service.chat(conversation_id=first["conversation_id"], question="金额较小呢？")

    assert second["conversation_id"] == first["conversation_id"]
    assert "历史轮数：2" in second["answer"]


def test_case_crud_and_memory_lookup():
    service = make_service()
    case = service.create_case(title="合同纠纷咨询", case_no="2026-民初-0428号", case_type="合同纠纷")
    chat = service.chat(case_id=case["id"], question="合同解除风险如何判断？")

    cases = service.list_cases()
    memory = service.get_case_memory(case["id"])
    deleted = service.delete_case(case["id"])

    assert cases[0]["title"] == "合同纠纷咨询"
    assert chat["case_id"] == case["id"]
    assert memory["case_id"] == case["id"]
    assert deleted["deleted"] is True
    assert service.list_cases() == []


def test_recent_history_keeps_chronological_pairs_and_truncates_long_content():
    service = make_service()
    service.history_limit = 2
    service.max_context_chars = 12
    first = service.chat(question="第一轮问题内容很长很长")
    service.chat(conversation_id=first["conversation_id"], question="第二轮问题")
    third = service.chat(conversation_id=first["conversation_id"], question="第三轮问题")

    conversation = service.get_conversation(third["conversation_id"])
    with service.session_factory() as session:
        history = service._recent_history(session, conversation["id"])

    assert [item["role"] for item in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"].startswith("第二轮问题")
    assert history[-1]["content"].endswith("...")
    assert all(len(item["content"]) <= 15 for item in history)
