import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import LegalAgent


class FakeRAG:
    def __init__(self, docs=None, answer="根据参考案例[1]，该行为可能涉及盗窃罪。"):
        self.docs = docs or []
        self.answer = answer
        self.last_top_k = None

    def retrieve(self, question, k=None):
        self.last_top_k = k
        return self.docs

    def generate_answer(self, question, docs):
        return self.answer


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def test_analyzes_question_intent_and_strategy():
    agent = LegalAgent(rag=FakeRAG())

    analysis = agent.analyze_question("有人入室盗窃，类似案例会怎么判？")

    assert analysis.intent == "similar_case"
    assert analysis.top_k == 5
    assert "案例" in analysis.reason


def test_rejects_non_legal_question_without_retrieval():
    rag = FakeRAG()
    agent = LegalAgent(rag=rag)

    result = agent.chat("帮我写一首关于春天的诗")

    assert result["intent"] == "out_of_scope"
    assert result["confidence"] == "low"
    assert result["references"] == []
    assert "法律" in result["answer"]
    assert rag.last_top_k is None


def test_returns_structured_answer_with_references_and_metadata():
    docs = [
        FakeDoc(
            "案情：被告人秘密窃取他人财物。\n罪名：盗窃罪\n相关法条：第264条\n刑期：1年",
            {"accusations": "盗窃罪", "articles": "[264]", "punishment": 1},
        )
    ]
    agent = LegalAgent(rag=FakeRAG(docs=docs))

    result = agent.chat("盗窃他人财物会承担什么法律责任？")

    assert result["intent"] == "legal_qa"
    assert result["confidence"] == "high"
    assert "仅供学习参考" in result["risk_notice"]
    assert result["references"][0]["id"] == 1
    assert result["references"][0]["accusations"] == "盗窃罪"
    assert result["references"][0]["articles"] == "[264]"
    assert "盗窃罪" in result["answer"]


def test_reports_insufficient_evidence_when_no_documents_found():
    agent = LegalAgent(rag=FakeRAG(docs=[]))

    result = agent.chat("这个非常冷门的问题应该怎么判断？")

    assert result["confidence"] == "low"
    assert result["references"] == []
    assert "无法根据知识库可靠回答" in result["answer"]
