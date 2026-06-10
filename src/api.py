import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))

from agent import LegalAgent
from conversation_service import ConversationService
from db import create_session_factory, init_db
from models import Base


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


def create_app(agent=None, conversation_service=None):
    app = FastAPI(
        title="中文法律问答 Agent",
        description="基于 RAG、Agent 工具调用和多轮记忆的中文法律知识库问答系统",
        version="1.0.0",
        docs_url="/api-docs",
        redoc_url=None,
    )
    legal_agent = agent
    service = conversation_service

    def get_agent():
        nonlocal legal_agent
        if legal_agent is None:
            legal_agent = LegalAgent()
        return legal_agent

    def get_service():
        nonlocal service
        if service is None:
            session_factory = create_session_factory()
            init_db(Base.metadata, session_factory)
            service = ConversationService(session_factory=session_factory, agent=get_agent())
        return service

    @app.post("/chat")
    def chat(request: ChatRequest):
        if conversation_service is not None or request.conversation_id or request.case_id:
            return get_service().chat(
                question=request.question,
                conversation_id=request.conversation_id,
                case_id=request.case_id,
            )
        if agent is not None:
            return get_agent().chat(request.question)
        return get_service().chat(question=request.question)

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        return {"question": request.question, "results": get_agent().retrieve(request.question, request.top_k)}

    @app.get("/cases")
    def list_cases():
        return get_service().list_cases()

    @app.post("/cases")
    def create_case(request: CaseCreateRequest):
        return get_service().create_case(title=request.title, case_no=request.case_no, case_type=request.case_type)

    @app.delete("/cases/{case_id}")
    def delete_case(case_id: str):
        return get_service().delete_case(case_id)

    @app.get("/conversations")
    def list_conversations(case_id: Optional[str] = None):
        return get_service().list_conversations(case_id=case_id)

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        return get_service().get_conversation(conversation_id)

    @app.get("/cases/{case_id}/memory")
    def get_case_memory(case_id: str):
        return get_service().get_case_memory(case_id)

    @app.get("/health")
    def health():
        try:
            session_factory = create_session_factory()
            with session_factory() as session:
                session.execute(text("select 1"))
            db_status = "ok"
        except Exception:
            db_status = "unavailable"

        if legal_agent is None:
            result = LegalAgent(rag=object()).health()
        else:
            result = get_agent().health()
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


DEMO_HTML = """
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
          <button class="small-btn" id="newCaseButton">新建</button>
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
          <button class="ghost-btn" id="retrieveButton">只检索证据</button>
        </div>
        <div class="conversation" id="conversation">
          <article class="message">
            <h4>用户问题</h4>
            <p id="questionPreview">盗窃他人财物会承担什么法律责任？</p>
          </article>
          <article class="message ai">
            <div class="risk" id="riskNotice">等待分析。回答将优先基于知识库案例和可追溯引用。</div>
            <h4>Agent 回答</h4>
            <div id="answerBox">输入案情或法律问题后，点击“生成法律分析”。</div>
          </article>
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
          <button class="primary-btn" id="sendButton">生成法律分析</button>
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
    const questionPreview = document.getElementById('questionPreview');
    const answerBox = document.getElementById('answerBox');
    const riskNotice = document.getElementById('riskNotice');
    const skeleton = document.getElementById('skeleton');
    const referencesBox = document.getElementById('references');
    const logicChain = document.getElementById('logicChain');
    const caseList = document.getElementById('caseList');
    const memoryCard = document.getElementById('memoryCard');
    let currentCaseId = null;
    let currentConversationId = null;

    document.getElementById('sendButton').addEventListener('click', runChat);
    document.getElementById('retrieveButton').addEventListener('click', runRetrieve);
    document.getElementById('newCaseButton').addEventListener('click', createCase);
    questionInput.addEventListener('input', syncQuestion);

    async function runChat() {
      syncQuestion();
      if (!questionInput.value.trim()) return;
      setLoading(true);
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
        renderChat(await response.json());
      } catch (error) {
        renderError(error);
      } finally {
        setLoading(false);
      }
    }

    async function runRetrieve() {
      syncQuestion();
      if (!questionInput.value.trim()) return;
      setLoading(true);
      try {
        const response = await fetch('/retrieve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: questionInput.value.trim(), top_k: 5 })
        });
        const data = await response.json();
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
      currentCaseId = data.case_id || currentCaseId;
      currentConversationId = data.conversation_id || currentConversationId;
      riskNotice.textContent = data.risk_notice || '回答仅供学习参考，不构成正式法律意见。';
      answerBox.innerHTML = `
        <p><strong>意图：</strong>${escapeHtml(data.intent || 'unknown')} · <strong>置信度：</strong>${escapeHtml(data.confidence || 'unknown')}</p>
        <p>${escapeHtml(data.answer || '暂无回答')}</p>
      `;
      renderReferences(data.references || []);
      renderLogic(data.steps || []);
      renderMemory(data.memory || {});
      if (currentCaseId) loadCases();
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
            <strong>${escapeHtml(item.case_no || '【案卷】')} ${escapeHtml(item.title)}</strong>
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
      questionPreview.textContent = questionInput.value.trim() || '请输入法律问题。';
    }

    function setLoading(active) {
      skeleton.classList.toggle('active', active);
    }

    function renderError(error) {
      riskNotice.textContent = '请求失败';
      answerBox.textContent = error.message;
    }

    function truncate(text, max) {
      return text.length > max ? `${text.slice(0, max)}...` : text;
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    }

    syncQuestion();
    loadHealth();
    loadCases();
  </script>
</body>
</html>
"""
