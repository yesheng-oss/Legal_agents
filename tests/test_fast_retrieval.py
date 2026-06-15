import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import LegalRAG


class FakeSessionScope:
    def __init__(self, session_factory):
        self.session = object()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_fast_retrieval_skips_query_rewrite_and_rerank(monkeypatch):
    class ExplodingRewriter:
        def rewrite(self, question):
            raise AssertionError("fast mode should not rewrite the question")

    class ExplodingReranker:
        def rerank(self, question, docs, top_k=3):
            raise AssertionError("fast mode should not rerank candidates")

    class FakeVectorRetriever:
        def retrieve(self, question, session, top_k=10, metadata_filter=None):
            assert question == "打架斗殴负什么责任"
            return [
                {
                    "id": 1,
                    "content": "罪名：故意伤害\n相关法条：第234条",
                    "accusations": "故意伤害",
                    "articles": "[234]",
                    "punishment": 12,
                }
            ]

    class FakeKeywordRetriever:
        def retrieve(self, question, session, top_k=10, metadata_filter=None):
            assert question == "打架斗殴负什么责任"
            return [
                {
                    "id": 2,
                    "content": "罪名：寻衅滋事\n相关法条：第293条",
                    "accusations": "寻衅滋事",
                    "articles": "[293]",
                    "punishment": 6,
                }
            ]

    monkeypatch.setenv("DEMO_FAST_MODE", "true")
    monkeypatch.setenv("SKIP_QUERY_REWRITE", "true")
    monkeypatch.setenv("SKIP_RERANK", "true")
    monkeypatch.setenv("FAST_RETRIEVAL_TOP_K", "1")
    monkeypatch.setattr("rag.create_session_factory", lambda: object())
    monkeypatch.setattr("rag.session_scope", FakeSessionScope)

    rag = LegalRAG.__new__(LegalRAG)
    rag.query_rewriter = ExplodingRewriter()
    rag.vector_retriever = FakeVectorRetriever()
    rag.keyword_retriever = FakeKeywordRetriever()
    rag.reranker = ExplodingReranker()

    docs = rag.retrieve("打架斗殴负什么责任")

    assert len(docs) == 1
    assert docs[0].page_content.startswith("罪名：故意伤害")
    assert docs[0].metadata["accusations"] == "故意伤害"


def test_full_retrieval_still_uses_query_rewrite_and_rerank(monkeypatch):
    calls = []

    class FakeRewriter:
        def rewrite(self, question):
            calls.append("rewrite")
            return "故意伤害 寻衅滋事"

    class FakeVectorRetriever:
        def retrieve(self, question, session, top_k=10, metadata_filter=None):
            assert question == "故意伤害 寻衅滋事"
            return [
                {
                    "id": 1,
                    "content": "罪名：故意伤害",
                    "accusations": "故意伤害",
                    "articles": "[234]",
                    "punishment": 12,
                }
            ]

    class FakeKeywordRetriever:
        def retrieve(self, question, session, top_k=10, metadata_filter=None):
            assert question == "故意伤害 寻衅滋事"
            return []

    class FakeReranker:
        def rerank(self, question, docs, top_k=3):
            calls.append("rerank")
            return docs[:top_k]

    monkeypatch.setenv("DEMO_FAST_MODE", "false")
    monkeypatch.setenv("SKIP_QUERY_REWRITE", "false")
    monkeypatch.setenv("SKIP_RERANK", "false")
    monkeypatch.setattr("rag.create_session_factory", lambda: object())
    monkeypatch.setattr("rag.session_scope", FakeSessionScope)

    rag = LegalRAG.__new__(LegalRAG)
    rag.query_rewriter = FakeRewriter()
    rag.vector_retriever = FakeVectorRetriever()
    rag.keyword_retriever = FakeKeywordRetriever()
    rag.reranker = FakeReranker()

    docs = rag.retrieve("打架斗殴负什么责任")

    assert calls == ["rewrite", "rerank"]
    assert len(docs) == 1
    assert docs[0].metadata["articles"] == "[234]"
