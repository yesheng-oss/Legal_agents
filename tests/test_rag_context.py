import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import LegalRAG
from rag import QueryRewriter
from rag import VectorRetriever


def test_format_history_uses_chinese_role_labels_and_limits_long_items():
    rag = LegalRAG.__new__(LegalRAG)

    text = rag._format_history(
        [
            {"role": "user", "content": "合同解除风险" * 120},
            {"role": "assistant", "content": "需要结合催告记录"},
        ]
    )

    assert "用户：" in text
    assert "助手：" in text
    assert "user:" not in text
    assert "assistant:" not in text
    assert "..." in text
    assert len(text) < 900


def test_vector_retriever_casts_embedding_parameter_to_pgvector():
    class FakeVector:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    class FakeEmbeddingModel:
        def encode(self, question):
            return FakeVector()

    class FakeSession:
        def execute(self, sql, params):
            self.sql = str(sql)
            self.params = params
            return []

    session = FakeSession()
    retriever = VectorRetriever(embed_model=FakeEmbeddingModel())

    retriever.retrieve("打架斗殴责任", session, top_k=3)

    assert "CAST(:embedding AS vector)" in session.sql
    assert session.params["embedding"] == "[0.1,0.2,0.3]"


def test_query_rewriter_keeps_legal_synonyms_when_llm_rewrite_is_unusable():
    class FakeLLM:
        def generate(self, prompt):
            return "无法识别的法律问题"

    rewritten = QueryRewriter(FakeLLM()).rewrite("打架斗殴负什么责任")

    assert "故意伤害" in rewritten
    assert "寻衅滋事" in rewritten
    assert "聚众斗殴" in rewritten
