# 法律案例 RAG 检索问答系统

这是一个面向中文法律问答、类案检索和案卷记忆的 RAG 检索问答系统。项目使用真实中文法律案例数据构建知识库，基于 PostgreSQL + pgvector 做语义检索，结合 PostgreSQL `pg_trgm` 关键词检索、多轮会话和流式输出，提供一个可本地演示的法律案例问答界面。

> 本项目用于学习、作品集和面试演示，回答仅供参考，不构成正式法律意见。

## 项目亮点

- **演示数据集**：原始数据来源为 Hugging Face `SunSpace0923/Refined-Chinese-Legal-Dataset`，支持 12 万级中文法律案例；当前仓库默认导入 1000 条案例用于本地演示。
- **RAG 检索链路**：案例文本切块后使用 `BAAI/bge-small-zh-v1.5` 生成 512 维向量，写入 PostgreSQL/pgvector。
- **混合检索**：同时支持 pgvector 向量召回和 PostgreSQL `pg_trgm` 关键词召回，并用 RRF 融合结果。
- **引用可追溯**：回答结果展示参考案例，同一案件的多个切片会合并为一个来源，并支持点击查看案例详情。
- **多轮会话**：支持案卷、会话、消息和案件长期记忆的 PostgreSQL 持久化。
- **演示界面**：FastAPI 内置前端页面，支持流式回答、证据来源、参考详情、案卷新建/重命名/删除。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| API | FastAPI / Uvicorn |
| RAG 编排 | LangChain Tool Calling + 检索兜底 |
| LLM | DeepSeek OpenAI-compatible API / Ollama |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| 向量库 | PostgreSQL + pgvector |
| 关键词检索 | PostgreSQL `pg_trgm` |
| ORM / 迁移 | SQLAlchemy + Alembic |
| 测试 | pytest / FastAPI TestClient |

## 启动方式选择

本项目保留两种启动方式：

- **Docker 部署**：适合面试官、同事或服务器快速复现，PostgreSQL/pgvector 由 Docker 提供。
- **本地无 Docker 开发/演示**：适合你自己电脑节省内存，只在本机运行 Python 服务，数据库使用本机或远程 PostgreSQL + pgvector。


## Docker 部署

### 1. 准备配置

复制 `.env.example` 为 `.env`，根据你的模型选择填写：

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b
```

如果使用 DeepSeek：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

### 2. 启动服务

```powershell
docker compose up -d postgres
docker compose up app
```

打开：

- 工作台：http://127.0.0.1:8000/docs
- Swagger API：http://127.0.0.1:8000/api-docs

Docker 路径会保留在 `docker-compose.yml` 中，适合对外部署和复现。

## 本地无 Docker 开发/演示

这种方式不启动 Docker。你需要自己准备一个支持 pgvector 的 PostgreSQL，可以是本机 PostgreSQL，也可以是 Neon、Supabase 等云端 PostgreSQL。

### 1. 安装 Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果项目已有 `.venv`，直接安装依赖即可：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 准备 PostgreSQL + pgvector

本机 PostgreSQL 示例：

```sql
CREATE USER legal_agent WITH PASSWORD 'legal_agent';
CREATE DATABASE legal_agent OWNER legal_agent;
\c legal_agent
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

`.env` 或当前 PowerShell 环境中配置：

```text
DATABASE_URL=postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
DEMO_FAST_MODE=true
SKIP_QUERY_REWRITE=true
SKIP_RERANK=true
FAST_RETRIEVAL_TOP_K=3
```

如果使用远程 PostgreSQL，把 `DATABASE_URL` 换成远程连接串即可。

### 3. 初始化本地数据库

如果你使用的是本仓库配置好的便携版 PostgreSQL，可以先启动它：

```powershell
.\scripts\start-portable-postgres.ps1
```

默认下载并导入 1000 条原始案例用于演示，避免首次处理 12 万条数据耗时过长，也保证普通电脑上展示更稳定：

```powershell
.\scripts\init-local-db.ps1
```

如果想调整下载和导入数量，可以同时设置 `DOWNLOAD_LIMIT` 和 `INGEST_LIMIT`：

```powershell
$env:DOWNLOAD_LIMIT=3000
$env:INGEST_LIMIT=3000
.\scripts\init-local-db.ps1
```

如果要完整导入，清空数量限制后重新下载并运行导入脚本：

```powershell
Remove-Item Env:\DOWNLOAD_LIMIT -ErrorAction SilentlyContinue
Remove-Item Env:\INGEST_LIMIT -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe src\download.py
.\.venv\Scripts\python.exe src\ingest.py
```

### 4. 启动本地服务

```powershell
.\scripts\start-local.ps1
```

或者手动启动：

```powershell
.\.venv\Scripts\uvicorn.exe src.api:app --host 127.0.0.1 --port 8000
```

打开：

- 工作台：http://127.0.0.1:8000/docs
- Swagger API：http://127.0.0.1:8000/api-docs

## 数据处理流程

1. `src/download.py` 默认下载 `train.json` 前 1000 条演示数据，可通过 `DOWNLOAD_LIMIT` 调整。
2. `src/ingest.py` 读取 `train.json`，默认按 `INGEST_LIMIT=1000` 导入演示案例，提取案情、罪名、法条和刑期。
3. 每条案例拼接成文本，并用 `RecursiveCharacterTextSplitter` 按 `CHUNK_SIZE=512`、`CHUNK_OVERLAP=64` 切块。
4. 每个 chunk 使用 `BAAI/bge-small-zh-v1.5` 生成 embedding。
5. chunk 文本、元数据、原始案件 ID 和向量写入 `legal_documents`。
6. 检索时同时走向量召回和关键词召回，再融合排序后交给 LLM 生成答案。
7. 前端证据来源会按原始案件 ID 合并同案切片，点击参考可查看完整案例详情。

## 主要接口

- `POST /chat`：普通法律问答，返回结构化答案。
- `POST /chat/stream`：流式法律问答，适合前端演示。
- `POST /retrieve`：只返回召回案例，方便展示 RAG 证据链。
- `GET /references/{source_case_id}`：查看同一原始案件的完整参考详情。
- `GET /cases` / `POST /cases` / `PATCH /cases/{case_id}` / `DELETE /cases/{case_id}`：案卷管理。
- `GET /conversations` / `GET /conversations/{conversation_id}`：会话管理。
- `GET /cases/{case_id}/memory`：查看案件长期记忆。
- `GET /health`：查看数据库、向量库和模型服务状态。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```


