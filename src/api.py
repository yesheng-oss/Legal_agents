import json
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).parent))

from agent import LegalAgent
from conversation_service import ConversationService
from db import create_session_factory, init_db, is_database_available
from models import Base
from settings import get_settings


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户法律问题")
    conversation_id: Optional[str] = Field(default=None, description="会话 ID")
    case_id: Optional[str] = Field(default=None, description="案卷 ID")


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="检索问题")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="返回案例数量")


class CaseCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="案卷标题")
    case_no: str = Field(default="", description="案号")
    case_type: str = Field(default="法律咨询", description="案由")


class CaseUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="案卷标题")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReferenceItem(BaseModel):
    id: int
    content: str
    accusations: str = ""
    articles: str = ""
    punishment: int = 0
    source_case_id: str = ""
    chunk_count: int = 1


class ReferenceDetailResponse(BaseModel):
    source_case_id: str
    accusations: str = ""
    articles: str = ""
    punishment: int = 0
    full_content: str = ""
    chunks: list[dict] = []


class ChatResponse(BaseModel):
    question: str
    intent: str
    answer: str
    confidence: str
    risk_notice: str = ""
    references: list[ReferenceItem] = []
    steps: list[str] = []
    case_id: Optional[str] = None
    conversation_id: Optional[str] = None
    memory: Optional[dict] = None


class RetrieveResponse(BaseModel):
    question: str
    results: list[ReferenceItem]


class CaseResponse(BaseModel):
    id: str
    title: str
    case_no: str = ""
    case_type: str = ""
    status: str = ""


class ConversationResponse(BaseModel):
    id: str
    case_id: str
    title: str


class MessageItem(BaseModel):
    id: str = ""
    role: str
    content: str
    model: str = ""
    references: list = []


class ConversationDetailResponse(BaseModel):
    id: str
    case_id: str
    title: str
    messages: list[MessageItem]


class CaseMemoryResponse(BaseModel):
    case_id: str
    facts_summary: str = ""
    user_goal: str = ""
    dispute_focus: str = ""
    confirmed_points: str = ""
    missing_evidence: str = ""


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str = ""
    ollama: str = ""


class DeleteResponse(BaseModel):
    deleted: bool


# ---------------------------------------------------------------------------
# Dependencies (module-level state, reset per create_app call)
# ---------------------------------------------------------------------------

_legal_agent: Optional[LegalAgent] = None
_conversation_service: Optional[ConversationService] = None


def get_agent() -> LegalAgent:
    global _legal_agent
    if _legal_agent is None:
        _legal_agent = LegalAgent()
    return _legal_agent


def get_service() -> ConversationService:
    if _conversation_service is None:
        if not is_database_available(timeout=1):
            raise HTTPException(
                status_code=503,
                detail="数据库未连接：请先启动 PostgreSQL/Docker，再重新点击按钮。",
            )
        session_factory = create_session_factory()
        init_db(Base.metadata, session_factory)
        return ConversationService(session_factory=session_factory, agent=get_agent())
    return _conversation_service


def warmup_retrieval_models():
    try:
        if not get_settings().warmup_retrieval:
            return
        agent = get_agent()
        vector_retriever = getattr(getattr(agent, "rag", None), "vector_retriever", None)
        embed_model = getattr(vector_retriever, "embed_model", None)
        if embed_model is not None:
            embed_model.encode("法律检索预热")
    except Exception:
        return


def get_reference_detail(source_case_id: str):
    session_factory = create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            text(
                """
                SELECT content, accusations, articles, punishment, source_chunk_index
                FROM legal_documents
                WHERE source_case_id = :source_case_id
                ORDER BY source_chunk_index ASC, id ASC
                """
            ),
            {"source_case_id": source_case_id},
        ).all()

    if not rows:
        raise KeyError(f"Reference not found: {source_case_id}")

    chunks = [
        {
            "index": row._mapping.get("source_chunk_index", index),
            "content": row._mapping.get("content", ""),
        }
        for index, row in enumerate(rows)
    ]
    first = rows[0]._mapping
    return {
        "source_case_id": source_case_id,
        "accusations": first.get("accusations", ""),
        "articles": first.get("articles", ""),
        "punishment": first.get("punishment", 0),
        "full_content": "\n\n".join(chunk["content"] for chunk in chunks if chunk["content"]),
        "chunks": chunks,
    }


def create_app(agent=None, conversation_service=None):
    global _legal_agent, _conversation_service
    _legal_agent = agent
    _conversation_service = conversation_service

    app = FastAPI(
        title="法律案例 RAG 检索问答系统",
        description="基于 RAG、混合检索和多轮案卷记忆的中文法律案例问答系统",
        version="1.0.0",
        docs_url="/api-docs",
        redoc_url=None,
    )

    @app.on_event("startup")
    async def warmup_on_startup():
        if _legal_agent is not None:
            return
        threading.Thread(target=warmup_retrieval_models, daemon=True).start()

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        agent_dep: LegalAgent = Depends(get_agent),
    ):
        try:
            if _conversation_service is not None or request.conversation_id or request.case_id:
                service_dep = get_service()
                result = await run_in_threadpool(
                    service_dep.chat,
                    question=request.question,
                    conversation_id=request.conversation_id,
                    case_id=request.case_id,
                )
                return result
            if _legal_agent is not None:
                result = await run_in_threadpool(agent_dep.chat, request.question)
                return result
            service_dep = get_service()
            result = await run_in_threadpool(service_dep.chat, question=request.question)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent执行出错：{exc}")

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        agent_dep: LegalAgent = Depends(get_agent),
    ):
        def sse(event: str, payload: dict) -> str:
            return format_sse(event, payload)

        def result_events(result: dict):
            yield sse(
                "meta",
                {
                    "question": result.get("question", request.question),
                    "intent": result.get("intent", "unknown"),
                    "confidence": result.get("confidence", "unknown"),
                    "risk_notice": result.get("risk_notice", ""),
                    "case_id": result.get("case_id"),
                    "conversation_id": result.get("conversation_id"),
                },
            )
            answer = result.get("answer", "")
            for index in range(0, len(answer), 48):
                yield sse("delta", {"text": answer[index : index + 48]})
            yield sse("references", {"references": result.get("references", [])})
            yield sse("memory", {"memory": result.get("memory") or {}})
            yield sse("steps", {"steps": result.get("steps", [])})
            yield sse("done", {"ok": True})

        def generate():
            try:
                if _conversation_service is not None or request.conversation_id or request.case_id:
                    service_dep = get_service()
                    if hasattr(service_dep, "chat_stream_events"):
                        try:
                            for event_name, payload in service_dep.chat_stream_events(
                                question=request.question,
                                conversation_id=request.conversation_id,
                                case_id=request.case_id,
                            ):
                                yield sse(event_name, payload)
                            return
                        except Exception:
                            result = service_dep.chat(
                                question=request.question,
                                conversation_id=request.conversation_id,
                                case_id=request.case_id,
                            )
                            yield from result_events(result)
                            return
                    result = service_dep.chat(
                        question=request.question,
                        conversation_id=request.conversation_id,
                        case_id=request.case_id,
                    )
                elif _legal_agent is not None:
                    result = agent_dep.chat(request.question)
                else:
                    service_dep = get_service()
                    if hasattr(service_dep, "chat_stream_events"):
                        try:
                            for event_name, payload in service_dep.chat_stream_events(question=request.question):
                                yield sse(event_name, payload)
                            return
                        except Exception:
                            result = service_dep.chat(question=request.question)
                            yield from result_events(result)
                            return
                    result = service_dep.chat(question=request.question)
                yield from result_events(result)
            except HTTPException as exc:
                yield sse("error", {"message": exc.detail})
            except Exception as exc:
                yield sse("error", {"message": f"Agent执行出错：{exc}"})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(
        request: RetrieveRequest,
        agent_dep: LegalAgent = Depends(get_agent),
    ):
        try:
            results = await run_in_threadpool(
                agent_dep.retrieve, request.question, request.top_k
            )
            return {"question": request.question, "results": results}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"检索出错：{exc}")

    @app.get("/cases", response_model=list[CaseResponse])
    async def list_cases(service_dep: ConversationService = Depends(get_service)):
        try:
            return await run_in_threadpool(service_dep.list_cases)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询案卷出错：{exc}")

    @app.post("/cases", response_model=CaseResponse)
    async def create_case(
        request: CaseCreateRequest,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(
                service_dep.create_case,
                title=request.title,
                case_no=request.case_no,
                case_type=request.case_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"创建案卷出错：{exc}")

    @app.patch("/cases/{case_id}", response_model=CaseResponse)
    async def rename_case(
        case_id: str,
        request: CaseUpdateRequest,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(
                service_dep.rename_case,
                case_id,
                request.title,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"重命名案卷出错：{exc}")

    @app.delete("/cases/{case_id}", response_model=DeleteResponse)
    async def delete_case(
        case_id: str,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(service_dep.delete_case, case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"删除案卷出错：{exc}")

    @app.get("/conversations", response_model=list[ConversationResponse])
    async def list_conversations(
        case_id: Optional[str] = None,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(service_dep.list_conversations, case_id=case_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询会话出错：{exc}")

    @app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
    async def get_conversation(
        conversation_id: str,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(
                service_dep.get_conversation, conversation_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询会话出错：{exc}")

    @app.get("/cases/{case_id}/memory", response_model=CaseMemoryResponse)
    async def get_case_memory(
        case_id: str,
        service_dep: ConversationService = Depends(get_service),
    ):
        try:
            return await run_in_threadpool(service_dep.get_case_memory, case_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询记忆出错：{exc}")

    @app.get("/references/{source_case_id}", response_model=ReferenceDetailResponse)
    async def reference_detail(source_case_id: str):
        try:
            return await run_in_threadpool(get_reference_detail, source_case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询参考详情出错：{exc}")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        db_available = await run_in_threadpool(is_database_available, timeout=1)
        db_status = "ok" if db_available else "unavailable"

        if not db_available:
            return {
                "status": "degraded",
                "database": db_status,
                "vector_store": "unavailable",
                "ollama": "not_checked",
            }

        if _legal_agent is None:
            result = LegalAgent(rag=object()).health()
        else:
            result = await run_in_threadpool(get_agent().health)
        result["database"] = db_status
        if db_status != "ok":
            result["status"] = "degraded"
        return result

    @app.get("/", response_class=HTMLResponse)
    @app.get("/docs", response_class=HTMLResponse)
    def demo():
        return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

    return app


app = create_app()

