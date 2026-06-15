$ErrorActionPreference = "Stop"

Write-Host "=== 法律案例 RAG 检索问答系统：本地数据库初始化（不启动 Docker） ===" -ForegroundColor Cyan

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$alembic = Join-Path $root ".venv\Scripts\alembic.exe"

if (-not (Test-Path $python)) {
    Write-Host "未找到 .venv，请先创建虚拟环境并安装依赖：" -ForegroundColor Red
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

if (-not $env:DATABASE_URL) {
    $env:DATABASE_URL = "postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent"
}

$limit = if ($env:INGEST_LIMIT) { $env:INGEST_LIMIT } else { "1000" }
$downloadLimit = if ($env:DOWNLOAD_LIMIT) { $env:DOWNLOAD_LIMIT } else { $limit }
Write-Host "DATABASE_URL=$env:DATABASE_URL"
Write-Host "DOWNLOAD_LIMIT=$downloadLimit"
Write-Host "INGEST_LIMIT=$limit"

Write-Host "[1/4] 检查数据库连接..."
@'
import sys
sys.path.insert(0, "src")
from db import is_database_available

raise SystemExit(0 if is_database_available(timeout=3) else 1)
'@ | & $python -
if ($LASTEXITCODE -ne 0) {
    Write-Host "数据库连接失败。请先启动本机 PostgreSQL，创建 legal_agent 数据库和用户，并启用 pgvector。" -ForegroundColor Red
    Write-Host "示例 SQL："
    Write-Host "  CREATE USER legal_agent WITH PASSWORD 'legal_agent';"
    Write-Host "  CREATE DATABASE legal_agent OWNER legal_agent;"
    Write-Host "  \c legal_agent"
    Write-Host "  CREATE EXTENSION IF NOT EXISTS vector;"
    Write-Host "  CREATE EXTENSION IF NOT EXISTS pg_trgm;"
    exit 1
}

Write-Host "[2/4] 迁移数据库..."
if (Test-Path $alembic) {
    & $alembic upgrade head
} else {
    & $python -m alembic upgrade head
}

Write-Host "[3/4] 下载数据（已存在则跳过）..."
$env:DOWNLOAD_LIMIT = $downloadLimit
& $python src\download.py

Write-Host "[4/4] 导入演示知识库..."
$env:INGEST_LIMIT = $limit
& $python src\ingest.py

Write-Host "本地数据库初始化完成。启动服务：" -ForegroundColor Green
Write-Host "  .\scripts\start-local.ps1"
