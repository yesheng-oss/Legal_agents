import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import create_app


class FakeAgent:
    def chat(self, question):
        return {
            "question": question,
            "intent": "legal_qa",
            "answer": "根据参考案例[1]，该问题需要结合具体事实分析。",
            "confidence": "high",
            "risk_notice": "回答仅供学习参考，不构成正式法律意见。",
            "references": [{"id": 1, "content": "案情：测试案例", "accusations": "测试罪名", "articles": "[1]", "punishment": 0}],
            "steps": ["问题分析", "案例检索", "答案生成", "引用校验"],
        }

    def retrieve(self, question, top_k=None):
        return [
            {
                "id": 1,
                "content": "案情：测试案例",
                "accusations": "测试罪名",
                "articles": "[1]",
                "punishment": 0,
            }
        ]

    def health(self):
        return {"status": "ok", "vector_store": "ok", "ollama": "unknown"}


class FakeConversationService:
    def chat(self, question, conversation_id=None, case_id=None):
        return {
            "question": question,
            "case_id": case_id or "case-1",
            "conversation_id": conversation_id or "conversation-1",
            "intent": "legal_qa",
            "answer": "多轮回答",
            "confidence": "high",
            "risk_notice": "回答仅供学习参考。",
            "references": [],
            "memory": {"case_id": case_id or "case-1", "facts_summary": "案件事实摘要"},
            "steps": ["问题分析"],
        }

    def create_case(self, title, case_no="", case_type="法律咨询"):
        return {"id": "case-1", "title": title, "case_no": case_no, "case_type": case_type, "status": "active"}

    def list_cases(self):
        return [{"id": "case-1", "title": "合同纠纷咨询", "case_no": "2026-民初-0428号", "case_type": "合同纠纷", "status": "active"}]

    def list_conversations(self, case_id=None):
        return [{"id": "conversation-1", "case_id": case_id or "case-1", "title": "盗窃罪怎么判？"}]

    def get_conversation(self, conversation_id):
        return {"id": conversation_id, "case_id": "case-1", "title": "标题", "messages": [{"role": "user", "content": "问题"}]}

    def get_case_memory(self, case_id):
        return {"case_id": case_id, "facts_summary": "案件事实摘要"}

    def delete_case(self, case_id):
        return {"deleted": True}


def test_chat_endpoint_returns_agent_result():
    client = TestClient(create_app(agent=FakeAgent()))

    response = client.post("/chat", json={"question": "盗窃罪怎么判？"})

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "legal_qa"
    assert data["references"][0]["id"] == 1


def test_chat_endpoint_supports_conversation_service_contract():
    client = TestClient(create_app(conversation_service=FakeConversationService()))

    response = client.post("/chat", json={"question": "金额较小呢？", "conversation_id": "conversation-1", "case_id": "case-1"})

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conversation-1"
    assert data["case_id"] == "case-1"
    assert data["memory"]["facts_summary"] == "案件事实摘要"


def test_case_conversation_and_memory_endpoints():
    client = TestClient(create_app(conversation_service=FakeConversationService()))

    assert client.post("/cases", json={"title": "合同纠纷咨询", "case_no": "2026-民初-0428号", "case_type": "合同纠纷"}).json()["id"] == "case-1"
    assert client.get("/cases").json()[0]["title"] == "合同纠纷咨询"
    assert client.get("/conversations", params={"case_id": "case-1"}).json()[0]["id"] == "conversation-1"
    assert client.get("/conversations/conversation-1").json()["messages"][0]["role"] == "user"
    assert client.get("/cases/case-1/memory").json()["facts_summary"] == "案件事实摘要"
    assert client.delete("/cases/case-1").json()["deleted"] is True


def test_retrieve_endpoint_returns_reference_list():
    client = TestClient(create_app(agent=FakeAgent()))

    response = client.post("/retrieve", json={"question": "找盗窃类案", "top_k": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["accusations"] == "测试罪名"


def test_health_endpoint_returns_component_status():
    client = TestClient(create_app(agent=FakeAgent()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "degraded")
    assert "database" in response.json()


def test_docs_serves_clean_legal_workbench_interface():
    client = TestClient(create_app(agent=FakeAgent()))

    response = client.get("/docs")

    assert response.status_code == 200
    assert "Legal AI Workbench" in response.text
    assert "legal-shell" in response.text
    assert "case-sidebar" in response.text
    assert "conversation-panel" in response.text
    assert "evidence-panel" in response.text
    assert "id=\"questionInput\"" in response.text
    assert "id=\"caseList\"" in response.text
    assert "id=\"memoryCard\"" in response.text
    assert "案件记忆" in response.text
    assert "证据来源" in response.text
    assert "数据仅在授权环境中处理" in response.text
    assert "fetch('/chat'" in response.text
    assert "fetch('/cases'" in response.text
    assert "case_id: currentCaseId" in response.text
    assert "conversation_id: currentConversationId" in response.text


def test_swagger_remains_available_for_api_debugging():
    client = TestClient(create_app(agent=FakeAgent()))

    response = client.get("/api-docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_chat_without_built_vector_store_returns_structured_notice():
    client = TestClient(create_app(conversation_service=FakeConversationService()))

    response = client.post("/chat", json={"question": "盗窃罪怎么判？"})

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conversation-1"
    assert data["case_id"] == "case-1"
    assert data["confidence"] == "high"
    assert data["references"] == []
    assert data["memory"]["facts_summary"] == "案件事实摘要"
