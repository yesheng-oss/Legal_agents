import json
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).parent))

from agent import LegalAgent
from conversation_service import ConversationService
from db import create_session_factory, init_db, is_database_available
from models import Base
from settings import get_settings


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


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReferenceItem(BaseModel):
    id: int
    content: str
    accusations: str = ""
    articles: str = ""
    punishment: int = 0


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


def create_app(agent=None, conversation_service=None):
    global _legal_agent, _conversation_service
    _legal_agent = agent
    _conversation_service = conversation_service

    app = FastAPI(
        title="中文法律问答 Agent",
        description="基于 RAG、Agent 工具调用和多轮记忆的中文法律知识库问答系统",
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
            return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

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

        return StreamingResponse(generate(), media_type="text/event-stream")

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
        return DEMO_HTML

    return app


app = create_app()


DEMO_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Legal AI Workbench</title>
  <style>
    :root {
      --navy: #0f172a;
      --blue: #1e3a8a;
      --amber: #d97706;
      --paper: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dde7;
      --ink: #111827;
      --muted: #667085;
      --soft: #eef2f7;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    #codex-browser-sidebar-comments-root,
    #codex-browser-sidebar-comments-root * {
      display: none !important;
      pointer-events: none !important;
    }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    button, textarea, input { font: inherit; }
    button { cursor: pointer; }
    .legal-shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(440px, 1fr) 340px;
    }
    .case-sidebar {
      background: var(--navy);
      color: #e5e7eb;
      padding: 22px 18px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .brand {
      border-bottom: 1px solid rgba(255,255,255,.12);
      padding-bottom: 18px;
    }
    .brand-mark {
      width: 40px;
      height: 40px;
      border: 1px solid rgba(255,255,255,.22);
      display: grid;
      place-items: center;
      color: #f8fafc;
      font-weight: 800;
      margin-bottom: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      color: #fff;
      letter-spacing: 0;
    }
    .brand p {
      margin: 7px 0 0;
      color: #a7b0c0;
      font-size: 12px;
      line-height: 1.6;
    }
    .side-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .side-head h2, .panel-title {
      margin: 0;
      font-size: 13px;
      font-weight: 800;
      color: inherit;
    }
    .small-btn, .primary-btn, .ghost-btn {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--navy);
      border-radius: 6px;
      padding: 9px 11px;
      font-weight: 700;
    }
    .small-btn {
      border-color: rgba(255,255,255,.18);
      background: rgba(255,255,255,.08);
      color: #f8fafc;
      padding: 6px 9px;
      font-size: 12px;
    }
    .primary-btn {
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }
    .case-list {
      display: grid;
      gap: 8px;
    }
    .case-item {
      width: 100%;
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.05);
      color: #e5e7eb;
      text-align: left;
      border-radius: 6px;
      padding: 11px;
    }
    .case-item.active {
      border-color: rgba(217,119,6,.72);
      background: rgba(217,119,6,.13);
    }
    .case-item strong {
      display: block;
      font-size: 13px;
      line-height: 1.45;
      color: #fff;
    }
    .case-item span {
      display: block;
      margin-top: 4px;
      color: #a7b0c0;
      font-size: 12px;
    }
    .security-note {
      margin-top: auto;
      color: #b8c2d3;
      font-size: 12px;
      line-height: 1.7;
      border-top: 1px solid rgba(255,255,255,.12);
      padding-top: 14px;
    }
    .conversation-panel {
      min-width: 0;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .topbar, .chat-card, .composer, .evidence-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .topbar {
      padding: 17px 18px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    .topbar h2 {
      margin: 0;
      font-size: 22px;
      color: var(--navy);
      letter-spacing: 0;
    }
    .topbar p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    .status-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      background: #fff;
      color: #475467;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .chat-card {
      flex: 1;
      min-height: 430px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .chat-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: #fbfcfe;
    }
    .chat-head h3 {
      margin: 0;
      font-size: 16px;
      color: var(--navy);
    }
    .conversation {
      flex: 1;
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 14px;
      background: #fff;
    }
    .message {
      border: 1px solid var(--line);
      border-left: 3px solid var(--blue);
      border-radius: 7px;
      padding: 14px;
      background: #fff;
      line-height: 1.75;
    }
    .message.ai {
      border-left-color: var(--amber);
      background: #fffdf8;
    }
    .message h4 {
      margin: 0 0 8px;
      color: var(--navy);
      font-size: 14px;
    }
    .answer-meta {
      margin: 0 0 12px;
      color: #344054;
      font-size: 13px;
    }
    .answer-content {
      line-height: 1.85;
      color: var(--ink);
      font-size: 15px;
    }
    .answer-content h2, .answer-content h3 {
      margin: 16px 0 8px;
      color: var(--navy);
      line-height: 1.35;
    }
    .answer-content h2 {
      font-size: 19px;
      padding-bottom: 7px;
      border-bottom: 1px solid var(--line);
    }
    .answer-content h3 { font-size: 16px; }
    .answer-content p { margin: 8px 0; }
    .answer-content ul, .answer-content ol {
      margin: 8px 0 10px 20px;
      padding: 0;
    }
    .answer-content li { margin: 5px 0; }
    .answer-content .citation {
      display: inline-grid;
      place-items: center;
      min-width: 24px;
      height: 24px;
      border-radius: 999px;
      background: #e8efff;
      color: var(--blue);
      font-weight: 800;
      font-size: 12px;
      margin: 0 2px;
    }
    .risk {
      border: 1px solid #f2c36b;
      background: #fffbeb;
      color: #78350f;
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 10px;
      font-weight: 700;
    }
    .composer {
      padding: 14px;
    }
    textarea {
      width: 100%;
      min-height: 108px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      color: var(--ink);
      outline: none;
      line-height: 1.65;
      background: #fff;
    }
    textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(30,58,138,.10);
    }
    .composer-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }
    .evidence-panel {
      border-left: 1px solid var(--line);
      background: #fbfcfe;
      padding: 22px 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .evidence-card {
      padding: 15px;
    }
    .evidence-card h3 {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--navy);
    }
    .reference-list {
      display: grid;
      gap: 10px;
    }
    .reference-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px;
      background: #fff;
      line-height: 1.65;
    }
    .reference-item strong {
      color: var(--navy);
      font-size: 13px;
    }
    .reference-item p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .memory-grid {
      display: grid;
      gap: 8px;
      color: #344054;
      font-size: 13px;
      line-height: 1.65;
    }
    .memory-grid strong {
      display: block;
      color: var(--navy);
      margin-bottom: 2px;
    }
    .steps {
      display: grid;
      gap: 8px;
    }
    .step {
      display: flex;
      gap: 9px;
      align-items: flex-start;
      color: #344054;
      font-size: 13px;
    }
    .step span {
      width: 22px;
      height: 22px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--soft);
      color: var(--blue);
      font-weight: 800;
      flex: 0 0 auto;
    }
    .skeleton {
      display: none;
      gap: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
    }
    .skeleton.active { display: grid; }
    .s-line {
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #eef2f7, #dfe5ee, #eef2f7);
      animation: pulse 1.2s infinite;
    }
    .s-line.short { width: 58%; }
    @keyframes pulse { 50% { opacity: .55; } }
    @media (max-width: 1120px) {
      .legal-shell { grid-template-columns: 240px 1fr; }
      .evidence-panel { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--line); }
    }
    @media (max-width: 760px) {
      .legal-shell { display: block; }
      .case-sidebar { min-height: auto; }
      .conversation-panel, .evidence-panel { padding: 14px; }
      .topbar { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="legal-shell">
    <aside class="case-sidebar">
      <section class="brand">
        <div class="brand-mark">LAW</div>
        <h1>Legal AI Workbench</h1>
        <p>面向法律问答、类案检索和案卷记忆的本地智能体工作台。</p>
      </section>

      <section>
        <div class="side-head">
          <h2>案卷</h2>
          <button class="small-btn" id="newCaseButton" onclick="createCase()" onpointerdown="createCase()">新建</button>
        </div>
        <div class="case-list" id="caseList">
          <button class="case-item active">
            <strong>【案号】2026-民初-0428号</strong>
            <span>合同纠纷咨询 · 示例案卷</span>
          </button>
        </div>
      </section>

      <p class="security-note">数据仅在授权环境中处理。请勿输入未脱敏的身份证号、银行卡号及其他敏感个人信息。</p>
    </aside>

    <main class="conversation-panel">
      <header class="topbar">
        <div>
          <h2>法律问答与案例检索</h2>
          <p>系统会结合最近多轮对话、案件记忆和向量库证据生成回答；依据不足时应提示补证。</p>
        </div>
        <div class="status-row">
          <span class="status-pill">向量库：<b id="vectorStatus">检测中</b></span>
          <span class="status-pill">模型：<b id="ollamaStatus">检测中</b></span>
          <a class="status-pill" href="/api-docs">Swagger</a>
        </div>
      </header>

      <section class="chat-card">
        <div class="chat-head">
          <h3>当前会话</h3>
        <button class="ghost-btn" id="retrieveButton" onclick="runRetrieve()" onpointerdown="runRetrieve()">只检索证据</button>
        </div>
        <div class="conversation" id="conversation">
          <div class="skeleton" id="skeleton">
            <div class="s-line"></div>
            <div class="s-line"></div>
            <div class="s-line short"></div>
          </div>
        </div>
      </section>

      <section class="composer">
        <textarea id="questionInput">盗窃他人财物会承担什么法律责任？</textarea>
        <div class="composer-actions">
          <span class="hint">建议输入：案情事实、金额、证据、诉求、已知争议点。</span>
          <button class="primary-btn" id="sendButton" onclick="runChat()" onpointerdown="runChat()" onmousedown="runChat()">生成法律分析</button>
        </div>
      </section>
    </main>

    <aside class="evidence-panel">
      <section class="evidence-card">
        <h3>证据来源</h3>
        <div class="reference-list" id="references">
          <div class="reference-item">
            <strong>暂无引用</strong>
            <p>运行检索或问答后，这里会展示向量库召回的案例、罪名、法条和摘要。</p>
          </div>
        </div>
      </section>

      <section class="evidence-card">
        <h3>案件记忆</h3>
        <div class="memory-grid" id="memoryCard">
          <div><strong>事实摘要</strong>暂无案件事实摘要。</div>
          <div><strong>争议焦点</strong>等待多轮对话提取。</div>
          <div><strong>待补充证据</strong>暂无补证建议。</div>
        </div>
      </section>

      <section class="evidence-card">
        <h3>推理步骤</h3>
        <div class="steps" id="logicChain">
          <div class="step"><span>1</span><div>问题分析</div></div>
          <div class="step"><span>2</span><div>案例检索</div></div>
          <div class="step"><span>3</span><div>回答生成与引用校验</div></div>
        </div>
      </section>
    </aside>
  </div>

  <script>
    const questionInput = document.getElementById('questionInput');
    const conversationBox = document.getElementById('conversation');
    const skeleton = document.getElementById('skeleton');
    const referencesBox = document.getElementById('references');
    const logicChain = document.getElementById('logicChain');
    const caseList = document.getElementById('caseList');
    const memoryCard = document.getElementById('memoryCard');
    let questionPreview = null;
    let answerBox = null;
    let riskNotice = null;
    let currentCaseId = null;
    let currentConversationId = null;
    let chatInFlight = false;

    function initLegalWorkbench() {
      const sendButton = document.getElementById('sendButton');
      const retrieveButton = document.getElementById('retrieveButton');
      const newCaseButton = document.getElementById('newCaseButton');
      window.runChat = runChat;
      window.runRetrieve = runRetrieve;
      window.createCase = createCase;
      sendButton.onclick = runChat;
      sendButton.onpointerdown = runChat;
      sendButton.onmousedown = runChat;
      retrieveButton.onclick = runRetrieve;
      retrieveButton.onpointerdown = runRetrieve;
      newCaseButton.onclick = createCase;
      newCaseButton.onpointerdown = createCase;
      questionInput.oninput = syncQuestion;
    }

    async function runChat() {
      if (!questionInput.value.trim()) return;
      if (chatInFlight) return;
      chatInFlight = true;
      prepareConversation();
      syncQuestion();
      setLoading(true);
      riskNotice.textContent = '正在检索证据...';
      answerBox.innerHTML = '<p class="answer-meta"><strong>意图：</strong>分析中 · <strong>置信度：</strong>生成中</p><div class="answer-content" id="streamAnswer">正在生成法律分析...</div>';
      let streamText = '';
      try {
        const response = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: questionInput.value.trim(),
            case_id: currentCaseId,
            conversation_id: currentConversationId
          })
        });
        if (!response.ok) throw new Error(`请求失败：${response.status}`);
        await readChatStream(response, (event, payload) => {
          if (event === 'meta') {
            currentCaseId = payload.case_id || currentCaseId;
            currentConversationId = payload.conversation_id || currentConversationId;
            riskNotice.textContent = payload.risk_notice || '正在生成法律分析...';
            answerBox.innerHTML = `
              <p class="answer-meta"><strong>意图：</strong>${escapeHtml(payload.intent || 'unknown')} · <strong>置信度：</strong>${escapeHtml(payload.confidence || 'unknown')}</p>
              <div class="answer-content" id="streamAnswer"></div>
            `;
          } else if (event === 'delta') {
            streamText += payload.text || '';
            const streamAnswer = document.getElementById('streamAnswer');
            if (streamAnswer) streamAnswer.innerHTML = renderMarkdown(streamText);
          } else if (event === 'references') {
            renderReferences(payload.references || []);
          } else if (event === 'memory') {
            renderMemory(payload.memory || {});
          } else if (event === 'steps') {
            renderLogic(payload.steps || []);
          } else if (event === 'error') {
            throw new Error(payload.message || '流式响应出错');
          }
        });
        if (currentCaseId) loadCases();
      } catch (error) {
        if (error.fromStreamEvent) {
          renderError(error);
        } else {
          await runChatFallback(error);
        }
      } finally {
        chatInFlight = false;
        setLoading(false);
      }
    }

    async function runChatFallback(originalError) {
      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: questionInput.value.trim(),
            case_id: currentCaseId,
            conversation_id: currentConversationId
          })
        });
        if (!response.ok) throw originalError;
        renderChat(await response.json());
      } catch (error) {
        renderError(error);
      }
    }

    async function runRetrieve() {
      if (!questionInput.value.trim()) return;
      prepareConversation();
      syncQuestion();
      setLoading(true);
      try {
        const response = await fetch('/retrieve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: questionInput.value.trim(), top_k: 5 })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `请求失败：${response.status}`);
        renderReferences(data.results || []);
        answerBox.textContent = `已检索到 ${(data.results || []).length} 条候选证据。`;
      } catch (error) {
        renderError(error);
      } finally {
        setLoading(false);
      }
    }

    async function createCase() {
      const title = questionInput.value.trim().slice(0, 24) || '新法律咨询';
      try {
        const response = await fetch('/cases', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, case_type: '法律咨询' })
        });
        const item = await response.json();
        currentCaseId = item.id;
        currentConversationId = null;
        await loadCases();
      } catch (error) {
        renderError(error);
      }
    }

    function renderChat(data) {
      prepareConversation();
      currentCaseId = data.case_id || currentCaseId;
      currentConversationId = data.conversation_id || currentConversationId;
      riskNotice.textContent = data.risk_notice || '回答仅供学习参考，不构成正式法律意见。';
      answerBox.innerHTML = `
        <p class="answer-meta"><strong>意图：</strong>${escapeHtml(data.intent || 'unknown')} · <strong>置信度：</strong>${escapeHtml(data.confidence || 'unknown')}</p>
        <div class="answer-content">${renderMarkdown(data.answer || '暂无回答')}</div>
      `;
      renderReferences(data.references || []);
      renderLogic(data.steps || []);
      renderMemory(data.memory || {});
      if (currentCaseId) loadCases();
    }

    async function readChatStream(response, onEvent) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split(String.fromCharCode(10, 10));
        buffer = events.pop() || '';
        events.forEach((rawEvent) => {
          const lines = rawEvent.split(String.fromCharCode(10));
          const eventLine = lines.find((line) => line.startsWith('event: '));
          const dataLine = lines.find((line) => line.startsWith('data: '));
          if (!eventLine || !dataLine) return;
          const event = eventLine.slice(7).trim();
          const payload = JSON.parse(dataLine.slice(6));
          if (event === 'error') {
            const error = new Error(payload.message || '流式响应出错');
            error.fromStreamEvent = true;
            throw error;
          }
          onEvent(event, payload);
        });
      }
    }

    function renderReferences(refs) {
      if (!refs.length) {
        referencesBox.innerHTML = '<div class="reference-item"><strong>暂无可引用证据</strong><p>知识库未返回足够材料时，系统应提示无法可靠判断。</p></div>';
        return;
      }
      referencesBox.innerHTML = refs.map((ref) => `
        <div class="reference-item">
          <strong>参考 ${escapeHtml(String(ref.id || '-'))} · ${escapeHtml(ref.accusations || '未标注罪名/案由')}</strong>
          <p>法条：${escapeHtml(ref.articles || '未标注')} · 刑期/结果：${escapeHtml(String(ref.punishment ?? '-'))}<br>${escapeHtml(truncate(ref.content || '', 150))}</p>
        </div>
      `).join('');
    }

    function renderLogic(steps) {
      const names = steps.length ? steps : ['问题分析', '案例检索', '回答生成与引用校验'];
      logicChain.innerHTML = names.map((step, index) => `
        <div class="step"><span>${index + 1}</span><div>${escapeHtml(step)}</div></div>
      `).join('');
    }

    function renderMemory(memory) {
      memoryCard.innerHTML = `
        <div><strong>事实摘要</strong>${escapeHtml(memory.facts_summary || '暂无案件事实摘要。')}</div>
        <div><strong>争议焦点</strong>${escapeHtml(memory.dispute_focus || '等待多轮对话提取。')}</div>
        <div><strong>待补充证据</strong>${escapeHtml(memory.missing_evidence || '暂无补证建议。')}</div>
      `;
    }

    async function loadHealth() {
      try {
        const response = await fetch('/health');
        const data = await response.json();
        document.getElementById('vectorStatus').textContent = data.vector_store || 'unknown';
        document.getElementById('ollamaStatus').textContent = data.model || data.ollama || 'unknown';
      } catch {
        document.getElementById('vectorStatus').textContent = 'unknown';
        document.getElementById('ollamaStatus').textContent = 'unknown';
      }
    }

    async function loadCases() {
      try {
        const response = await fetch('/cases');
        const cases = await response.json();
        if (!Array.isArray(cases) || !cases.length) return;
        currentCaseId = currentCaseId || cases[0].id;
        caseList.innerHTML = cases.map((item) => `
          <button class="case-item ${item.id === currentCaseId ? 'active' : ''}" data-case-id="${escapeHtml(item.id)}">
            <strong>${escapeHtml(item.case_no || '【案卷】')} ${escapeHtml(cleanTitle(item.title))}</strong>
            <span>${escapeHtml(item.case_type || '法律咨询')} · ${escapeHtml(item.status || 'active')}</span>
          </button>
        `).join('');
        caseList.querySelectorAll('.case-item').forEach((button) => {
          button.addEventListener('click', () => {
            currentCaseId = button.dataset.caseId;
            currentConversationId = null;
            loadCaseMemory(currentCaseId);
            loadCases();
          });
        });
        loadCaseMemory(currentCaseId);
      } catch {
        return;
      }
    }

    async function loadCaseMemory(caseId) {
      if (!caseId) return;
      try {
        const response = await fetch(`/cases/${caseId}/memory`);
        renderMemory(await response.json());
      } catch {
        return;
      }
    }

    function syncQuestion() {
      if (questionPreview) {
        questionPreview.textContent = questionInput.value.trim() || '请输入法律问题。';
      }
    }

    function setLoading(active) {
      skeleton.classList.toggle('active', active);
    }

    function renderError(error) {
      prepareConversation();
      riskNotice.textContent = '请求失败';
      answerBox.textContent = error.message;
    }

    function prepareConversation() {
      if (answerBox && riskNotice && questionPreview) return;
      conversationBox.querySelectorAll('.message').forEach((item) => item.remove());
      const questionArticle = document.createElement('article');
      questionArticle.className = 'message';
      questionArticle.innerHTML = '<h4>用户问题</h4><p id="questionPreview"></p>';
      const answerArticle = document.createElement('article');
      answerArticle.className = 'message ai';
      answerArticle.innerHTML = '<div class="risk" id="riskNotice">正在准备分析...</div><h4>Agent 回答</h4><div id="answerBox"></div>';
      conversationBox.insertBefore(questionArticle, skeleton);
      conversationBox.insertBefore(answerArticle, skeleton);
      questionPreview = document.getElementById('questionPreview');
      answerBox = document.getElementById('answerBox');
      riskNotice = document.getElementById('riskNotice');
    }

    function truncate(text, max) {
      return text.length > max ? `${text.slice(0, max)}...` : text;
    }

    function cleanTitle(title) {
      const value = String(title || '').trim();
      const questionMarks = (value.match(/\?/g) || []).length;
      const mojibake = /[ÃÂ�]/.test(value) || /[çäåæèéêëìíîïðñòóôõöùúûü]/i.test(value);
      if (!value || questionMarks >= 4 || mojibake) return '未命名案卷';
      return value;
    }

    function renderMarkdown(markdown) {
      const newline = String.fromCharCode(10);
      const lines = String(markdown || '').replace(/\r\n/g, newline).split(newline);
      const html = [];
      let listType = null;
      const closeList = () => {
        if (listType) {
          html.push(`</${listType}>`);
          listType = null;
        }
      };
      lines.forEach((rawLine) => {
        const line = rawLine.trim();
        if (!line) {
          closeList();
          return;
        }
        const heading = line.match(/^(#{2,3})\s+(.+)$/);
        if (heading) {
          closeList();
          const level = heading[1].length;
          html.push(`<h${level}>${formatInlineMarkdown(heading[2])}</h${level}>`);
          return;
        }
        const ordered = line.match(/^\d+[.、]\s+(.+)$/);
        if (ordered) {
          if (listType !== 'ol') {
            closeList();
            listType = 'ol';
            html.push('<ol>');
          }
          html.push(`<li>${formatInlineMarkdown(ordered[1])}</li>`);
          return;
        }
        const unordered = line.match(/^[-*]\s+(.+)$/);
        if (unordered) {
          if (listType !== 'ul') {
            closeList();
            listType = 'ul';
            html.push('<ul>');
          }
          html.push(`<li>${formatInlineMarkdown(unordered[1])}</li>`);
          return;
        }
        closeList();
        html.push(`<p>${formatInlineMarkdown(line)}</p>`);
      });
      closeList();
      return html.join('');
    }

    function formatInlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/【?参考案例?】?\[(\d+)\]|【(\d+)】|\[(\d+)\]/g, (_, a, b, c) => `<span class="citation">${a || b || c}</span>`);
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    }

    initLegalWorkbench();
    loadHealth();
    loadCases();
  </script>
</body>
</html>
"""
