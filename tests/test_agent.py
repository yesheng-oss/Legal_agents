import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent import LegalAgent, SYSTEM_PROMPT, analyze_question, check_citation


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


class FakeRAG:
    def __init__(self, docs=None, answer="根据参考案例[1]，该行为可能涉及盗窃罪。"):
        self.docs = docs or []
        self.answer = answer
        self.last_top_k = None

    def retrieve(self, question, k=None):
        self.last_top_k = k
        return self.docs

    def generate_answer(self, question, docs, history=None, memory=None):
        return self.answer


class FakeToolChatModel(BaseChatModel):
    """支持 tool-calling 的 Fake ChatModel，用于测试 Agent 图。"""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self._responses = responses
        self._idx = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self._responses[self._idx]
        self._idx += 1
        msg = AIMessage(
            content=response.get("content", ""),
            tool_calls=response.get("tool_calls", []),
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "fake_tool_chat"

    @property
    def _identifying_params(self):
        return {}


def _build_tools_for_agent(agent):
    """为指定 LegalAgent 实例构建绑定好的工具列表（与 agent._build_agent_graph 一致）。"""

    def _retrieve_legal_cases_bound(question: str, top_k: int = 3) -> str:
        """根据问题检索相关法律案例，返回 JSON 字符串。"""
        return agent._run_retrieve(question, top_k, agent=agent)

    _retrieve_legal_cases_bound.__name__ = "retrieve_legal_cases"

    return [
        analyze_question,
        tool(_retrieve_legal_cases_bound),
        check_citation,
    ]


def _make_agent_with_responses(rag, responses):
    """用 Fake ChatModel 构建一个可预测的 LegalAgent。"""
    agent = LegalAgent(rag=rag)
    tools = _build_tools_for_agent(agent)
    fake_model = FakeToolChatModel(responses=responses)
    agent._agent_graph = create_agent(
        model=fake_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def test_analyzes_question_intent_and_strategy():
    agent = LegalAgent(rag=FakeRAG())

    analysis = agent.analyze_question("有人入室盗窃，类似案例会怎么判？")

    assert analysis.intent == "similar_case"
    assert analysis.top_k == 5
    assert "案例" in analysis.reason


def test_analyzes_common_fight_question_as_legal_question():
    agent = LegalAgent(rag=FakeRAG())

    analysis = agent.analyze_question("打架斗殴负什么责任？")

    assert analysis.intent == "legal_qa"
    assert analysis.top_k == 3


def test_rejects_non_legal_question_without_retrieval():
    rag = FakeRAG()
    responses = [
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "analyze_question",
                    "args": {"question": "帮我写一首关于春天的诗"},
                }
            ]
        },
        {"content": "该问题超出中文法律知识库的回答范围。\n【置信度：low】"},
    ]
    agent = _make_agent_with_responses(rag, responses)

    result = agent.chat("帮我写一首关于春天的诗")

    assert result["intent"] == "out_of_scope"
    assert result["confidence"] == "low"
    assert result["references"] == []
    assert "法律" in result["answer"]
    assert rag.last_top_k is None


def test_falls_back_to_rag_when_llm_misclassifies_legal_question_as_out_of_scope():
    docs = [
        FakeDoc(
            "案情：双方因琐事发生厮打，被告人致被害人轻伤。\n罪名：故意伤害\n相关法条：第234条",
            {"accusations": "故意伤害", "articles": "[234]", "punishment": 10},
        )
    ]
    rag = FakeRAG(docs=docs, answer="打架斗殴可能涉及治安处罚、民事赔偿；造成轻伤以上可能构成故意伤害罪，参考案例[1]。")
    responses = [
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "analyze_question",
                    "args": {"question": "无法识别的法律问题"},
                }
            ]
        },
        {"content": "该问题超出中文法律知识库的回答范围。\n【置信度：low】"},
    ]
    agent = _make_agent_with_responses(rag, responses)

    result = agent.chat("打架斗殴负什么责任？")

    assert result["intent"] == "legal_qa"
    assert result["confidence"] == "high"
    assert result["references"][0]["accusations"] == "故意伤害"
    assert "参考案例[1]" in result["answer"]


def test_returns_structured_answer_with_references_and_metadata():
    docs = [
        FakeDoc(
            "案情：被告人秘密窃取他人财物。\n罪名：盗窃罪\n相关法条：第264条\n刑期：1年",
            {"accusations": "盗窃罪", "articles": "[264]", "punishment": 1},
        )
    ]
    rag = FakeRAG(docs=docs)
    ref_payload = json.dumps(
        {
            "references": [
                {
                    "id": 1,
                    "content": docs[0].page_content,
                    "accusations": "盗窃罪",
                    "articles": "[264]",
                    "punishment": 1,
                }
            ]
        },
        ensure_ascii=False,
    )
    responses = [
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "analyze_question",
                    "args": {"question": "盗窃他人财物会承担什么法律责任？"},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "retrieve_legal_cases",
                    "args": {"question": "盗窃他人财物会承担什么法律责任？", "top_k": 3},
                }
            ]
        },
        {
            "content": "根据参考案例[1]，盗窃他人财物可能构成盗窃罪。",
            "tool_calls": [
                {
                    "id": "call_3",
                    "name": "check_citation",
                    "args": {"answer": "根据参考案例[1]，盗窃他人财物可能构成盗窃罪。"},
                }
            ],
        },
        {"content": "根据参考案例[1]，盗窃他人财物可能构成盗窃罪，适用刑法第264条。\n【置信度：high】"},
    ]
    agent = _make_agent_with_responses(rag, responses)

    result = agent.chat("盗窃他人财物会承担什么法律责任？")

    assert result["intent"] == "legal_qa"
    assert result["confidence"] == "high"
    assert "仅供学习参考" in result["risk_notice"]
    assert result["references"][0]["id"] == 1
    assert result["references"][0]["accusations"] == "盗窃罪"
    assert result["references"][0]["articles"] == "[264]"
    assert "盗窃罪" in result["answer"]


def test_reports_insufficient_evidence_when_no_documents_found():
    rag = FakeRAG(docs=[])
    responses = [
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "analyze_question",
                    "args": {"question": "这个非常冷门的问题应该怎么判断？"},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "retrieve_legal_cases",
                    "args": {"question": "这个非常冷门的问题应该怎么判断？", "top_k": 3},
                }
            ]
        },
        {"content": "无法根据知识库可靠回答。\n【置信度：low】"},
    ]
    agent = _make_agent_with_responses(rag, responses)

    result = agent.chat("这个非常冷门的问题应该怎么判断？")

    assert result["confidence"] == "low"
    assert result["references"] == []
    assert "无法根据知识库可靠回答" in result["answer"]
