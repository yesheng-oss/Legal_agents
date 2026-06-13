from typing import Generator, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy import text

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KEYWORD_TOP_K,
    RERANK_MODEL,
    RERANK_TOP_K,
    RRF_K,
    TOP_K,
    VECTOR_TOP_K,
)
from db import create_session_factory, session_scope
from llm import create_llm_provider
from settings import get_settings


PROMPT = """你是一名中国法律知识检索增强问答助手。
请只基于"法律案例参考、历史对话、案件记忆"回答用户问题；如果依据不足，请明确说明无法可靠判断，并列出需要补充的证据。

历史对话：
{history}

案件记忆：
{memory}

法律案例参考：
{context}

用户问题：
{question}

请输出结构化中文法律分析，包含：核心结论、依据说明、参考案例、风险提示。"""


# ---------------------------------------------------------------------------
# Query Rewriter：把口语化问题改写成标准法律术语
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """你是一名法律检索优化助手。请将用户的口语化法律问题改写为更适合向量数据库检索的标准法律术语表述。
改写要求：
1. 使用正式法律术语（如"盗窃罪"而非"偷东西"）
2. 保留核心法律概念和关键事实
3. 只输出改写后的查询，不要解释

用户问题：{question}
改写："""


class QueryRewriter:
    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm_provider()
        return self._llm

    def rewrite(self, question: str) -> str:
        prompt = REWRITE_PROMPT.format(question=question)
        try:
            rewritten = self.llm.generate(prompt).strip()
        except Exception:
            rewritten = ""
        # 简单清理：去掉引号和多余前缀
        rewritten = rewritten.strip('"""').strip("'").strip()
        if rewritten.lower().startswith("改写："):
            rewritten = rewritten[3:].strip()
        if not rewritten or self._is_unusable_rewrite(rewritten):
            rewritten = question
        return self._expand_legal_synonyms(question, rewritten)

    @staticmethod
    def _is_unusable_rewrite(rewritten: str) -> bool:
        unusable_markers = ("无法识别", "无法判断", "不能识别", "不属于法律", "无法改写")
        return any(marker in rewritten for marker in unusable_markers)

    @staticmethod
    def _expand_legal_synonyms(question: str, rewritten: str) -> str:
        expansions = []
        if any(term in question for term in ("打架", "斗殴", "互殴")):
            expansions.extend(["故意伤害", "寻衅滋事", "聚众斗殴", "治安管理处罚", "民事赔偿", "刑事责任"])
        if "偷" in question and "盗窃" not in rewritten:
            expansions.append("盗窃")
        if "骗" in question and "诈骗" not in rewritten:
            expansions.append("诈骗")
        if not expansions:
            return rewritten
        merged = [rewritten]
        merged.extend(term for term in expansions if term not in rewritten)
        return " ".join(merged)


# ---------------------------------------------------------------------------
# Vector Retriever：pgvector 余弦相似度检索
# ---------------------------------------------------------------------------

class VectorRetriever:
    def __init__(self, embed_model=None):
        self._embed_model = embed_model

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embed_model

    def retrieve(
        self,
        question: str,
        session,
        top_k: int = VECTOR_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        embedding = self.embed_model.encode(question).tolist()
        embedding_param = "[" + ",".join(str(value) for value in embedding) + "]"

        where_clauses = []
        params: dict = {"embedding": embedding_param, "top_k": top_k}

        if metadata_filter:
            if metadata_filter.get("accusations"):
                where_clauses.append("accusations ILIKE :accusations")
                params["accusations"] = f"%{metadata_filter['accusations']}%"

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
        SELECT id, content, accusations, articles, punishment,
               1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM legal_documents
        {where_sql}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
        result = session.execute(text(sql), params)
        return [dict(row._mapping) for row in result]


# ---------------------------------------------------------------------------
# Keyword Retriever：pg_trgm 文本相似度检索
# ---------------------------------------------------------------------------

class KeywordRetriever:
    def retrieve(
        self,
        question: str,
        session,
        top_k: int = KEYWORD_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        where_clauses = []
        params: dict = {"query": question, "top_k": top_k}

        if metadata_filter:
            if metadata_filter.get("accusations"):
                where_clauses.append("accusations ILIKE :accusations")
                params["accusations"] = f"%{metadata_filter['accusations']}%"

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # 使用 trigram similarity 排序；不对 content 强制相似度阈值，
        # 避免短查询被过滤掉（1000 条数据性能可接受）
        sql = f"""
        SELECT id, content, accusations, articles, punishment,
               similarity(content, :query) AS score
        FROM legal_documents
        {where_sql}
        ORDER BY similarity(content, :query) DESC
        LIMIT :top_k
        """
        result = session.execute(text(sql), params)
        return [dict(row._mapping) for row in result]


# ---------------------------------------------------------------------------
# Reranker：Cross-Encoder 精排
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or RERANK_MODEL
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, question: str, docs: list[dict], top_k: int = TOP_K) -> list[dict]:
        if not docs:
            return []
        pairs = [[question, doc["content"]] for doc in docs]
        scores = self.model.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# RRF Fusion：Reciprocal Rank Fusion 融合排序
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """融合向量检索和关键词检索结果，去重后按 RRF 分数排序。"""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ---------------------------------------------------------------------------
# LegalRAG：组合检索 + 生成
# ---------------------------------------------------------------------------

class LegalRAG:
    def __init__(self, llm_provider=None, embed_model=None, reranker=None):
        self.llm_provider = llm_provider or create_llm_provider()
        self.query_rewriter = QueryRewriter(self.llm_provider)
        self.vector_retriever = VectorRetriever(embed_model)
        self.keyword_retriever = KeywordRetriever()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[Document]:
        """混合检索入口：改写 → 向量 + 关键词 → RRF 融合 → Cross-Encoder 精排。"""
        settings = get_settings()
        fast_mode = settings.demo_fast_mode
        rewritten = question if fast_mode and settings.skip_query_rewrite else self.query_rewriter.rewrite(question)

        session_factory = create_session_factory()
        with session_scope(session_factory) as session:
            vec_results = self.vector_retriever.retrieve(
                rewritten, session, top_k=VECTOR_TOP_K, metadata_filter=metadata_filter
            )
            key_results = self.keyword_retriever.retrieve(
                rewritten, session, top_k=KEYWORD_TOP_K, metadata_filter=metadata_filter
            )

        fused = reciprocal_rank_fusion(vec_results, key_results)
        target_k = k or (settings.fast_retrieval_top_k if fast_mode else TOP_K)
        if fast_mode and settings.skip_rerank:
            reranked = fused[:target_k]
        else:
            reranked = self.reranker.rerank(question, fused, top_k=target_k)

        # 返回 LangChain Document 以保持与 agent.py 兼容
        return [
            Document(
                page_content=doc["content"],
                metadata={
                    "accusations": doc.get("accusations", ""),
                    "articles": doc.get("articles", ""),
                    "punishment": doc.get("punishment", 0),
                },
            )
            for doc in reranked
        ]

    def generate_answer(
        self, question: str, docs: list[Document], history: Optional[list] = None, memory: Optional[dict] = None
    ) -> str:
        prompt = self._build_answer_prompt(question, docs, history=history, memory=memory)
        return self.llm_provider.generate(prompt)

    def generate_answer_stream(
        self, question: str, docs: list[Document], history: Optional[list] = None, memory: Optional[dict] = None
    ) -> Generator[str, None, None]:
        prompt = self._build_answer_prompt(question, docs, history=history, memory=memory)
        yield from self.llm_provider.generate_stream(prompt)

    def _build_answer_prompt(
        self, question: str, docs: list[Document], history: Optional[list] = None, memory: Optional[dict] = None
    ) -> str:
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        history_text = self._format_history(history or [])
        memory_text = self._format_memory(memory or {})
        return PROMPT.format(
            context=context, question=question, history=history_text, memory=memory_text
        )

    def query(self, question: str):
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
        return "\n".join(
            f"{labels.get(key, key)}：{self._trim_prompt_text(str(value))}"
            for key, value in memory.items()
            if value
        )

    def _trim_prompt_text(self, text: str, max_chars: int = 360) -> str:
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
