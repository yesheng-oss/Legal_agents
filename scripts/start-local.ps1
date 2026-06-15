$ErrorActionPreference = "Stop"

Write-Host "=== 法律案例 RAG 检索问答系统：本地启动（不启动 Docker） ===" -ForegroundColor Cyan

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "未找到 .venv，请先运行：" -ForegroundColor Red
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

if (-not $env:DATABASE_URL) {
    $env:DATABASE_URL = "postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent"
}
if (-not $env:LLM_PROVIDER) {
    $env:LLM_PROVIDER = "ollama"
}
if (-not $env:OLLAMA_BASE_URL) {
    $env:OLLAMA_BASE_URL = "http://localhost:11434"
}
if (-not $env:OLLAMA_MODEL) {
    $env:OLLAMA_MODEL = "qwen2.5:7b"
}
if (-not $env:DEMO_FAST_MODE) {
    $env:DEMO_FAST_MODE = "true"
}

Write-Host "DATABASE_URL=$env:DATABASE_URL"
Write-Host "LLM_PROVIDER=$env:LLM_PROVIDER"

@'
import sys
sys.path.insert(0, "src")
from db import is_database_available

raise SystemExit(0 if is_database_available(timeout=3) else 1)
'@ | & $python -
if ($LASTEXITCODE -ne 0) {
    Write-Host "数据库连接失败。请确认本机/远程 PostgreSQL 已启动，并已安装 pgvector。" -ForegroundColor Red
    Write-Host "可先运行：.\scripts\init-local-db.ps1"
    exit 1
}

@'
import sys
sys.path.insert(0, "src")
from db import create_session_factory, session_scope
from sqlalchemy import text

sf = create_session_factory()
with session_scope(sf) as session:
    count = session.execute(text("select count(*) from legal_documents")).scalar()
print(f"legal_documents={count}")
'@ | & $python -

Write-Host "启动服务：http://127.0.0.1:8000/docs" -ForegroundColor Green
& $python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
