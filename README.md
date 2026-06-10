# Legal RAG Agent — 中文法律知识库问答智能体

基于 **RAG（检索增强生成）+ Agent 工具调用流程** 的中文法律知识库问答系统。从 Hugging Face 下载真实法律案例，构建向量索引，结合 Ollama 本地大模型进行法律问答，并通过问题分析、案例检索、答案生成、引用校验输出可追溯的法律辅助回答。

## 功能

- 从 Hugging Face 自动下载中文法律数据集（12 万+ 条真实案例）
- 使用 `BAAI/bge-small-zh-v1.5` 语义嵌入模型构建 Chroma 向量索引
- 基于 Ollama `qwen2.5:7b` 进行法律知识问答
- Agent 工具链：问题意图识别、检索策略选择、案例检索、答案生成、引用校验
- FastAPI + Swagger 接口展示，支持结构化返回答案、参考案例、罪名、法条、置信度和风险提示
- 小型评测集覆盖法律问答、类案检索、量刑参考、非法律问题拒答和证据不足场景

## 快速开始

### 前置条件

- [Ollama](https://ollama.com) 已安装并运行
- Python 3.10+

### 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 拉取对话模型
ollama pull qwen2.5:7b
```

### 配置 DeepSeek 与 PostgreSQL

复制 `.env.example` 为 `.env`，至少填写：

```text
DATABASE_URL=postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
```

启动 PostgreSQL：

```bash
docker compose up -d postgres
```

初始化业务数据库表：

```bash
alembic upgrade head
```

如果暂时没有 DeepSeek API Key，可以把 `LLM_PROVIDER` 设置为 `ollama`，继续使用本地 `qwen2.5:7b`。

### 运行

```bash
# 方式一：一键启动
python src/app.py

# 方式二：分步执行
python src/download.py   # 下载数据
python src/ingest.py     # 构建索引
python src/rag.py        # 启动查询
```

Windows 下也可双击 `run.bat` 或运行 `run.ps1`。

### 启动 Agent API

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

打开法律 AI 工作台演示界面：

```text
http://127.0.0.1:8000/docs
```

界面包含三栏式工作台：左侧案卷与法规工具导航、中间智能对话与证据链、右侧法条/案例推荐、证据链图谱和法律意见书预览。

打开 Swagger API 文档：

```text
http://127.0.0.1:8000/api-docs
```

接口说明：

- `POST /chat`：输入法律问题，返回 Agent 最终回答、置信度、风险提示和参考案例。
- `GET /cases`：返回案卷列表，用于工作台左侧边栏。
- `POST /cases`：创建新案卷。
- `GET /conversations?case_id=...`：返回案卷下的会话列表。
- `GET /conversations/{conversation_id}`：返回完整多轮消息历史。
- `GET /cases/{case_id}/memory`：返回案件长期记忆摘要。
- `POST /retrieve`：只返回检索结果，便于展示 RAG 召回效果。
- `GET /health`：检查向量库和 Ollama 服务状态。

示例请求：

```json
{
  "question": "盗窃他人财物会承担什么法律责任？",
  "case_id": "可选案卷ID",
  "conversation_id": "可选会话ID"
}
```

## 项目结构

```
├── src/
│   ├── config.py      # 配置（模型、路径等）
│   ├── download.py    # Hugging Face 数据下载
│   ├── ingest.py      # 数据处理与索引构建
│   ├── rag.py         # RAG 查询引擎
│   ├── agent.py       # 法律问答 Agent 工具链
│   ├── llm.py         # DeepSeek / Ollama 模型适配层
│   ├── db.py          # SQLAlchemy 数据库会话
│   ├── models.py      # PostgreSQL ORM 模型
│   ├── memory.py      # 案件长期记忆服务
│   ├── conversation_service.py # 多轮会话与案卷服务
│   ├── api.py         # FastAPI / Swagger 接口
│   └── app.py         # CLI 交互入口
├── alembic/           # PostgreSQL 数据库迁移
├── tests/             # Agent 与 API 自动化测试
├── eval_cases.json    # 小型场景评测集
├── data/
│   ├── raw/           # 原始法律案例数据
│   └── chroma/        # Chroma 向量索引
├── requirements.txt
├── run.bat / run.ps1  # 一键启动脚本
└── README.md
```

## 数据来源

- **数据集**: [SunSpace0923/Refined-Chinese-Legal-Dataset](https://huggingface.co/datasets/SunSpace0923/Refined-Chinese-Legal-Dataset)
- **规模**: ~12.3 万条中文法律案例
- **字段**: 案情事实 (fact)、罪名 (accusation)、相关法条 (relevant_articles)、刑期 (term_of_imprisonment)

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量数据库 | Chroma |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 对话模型 | Qwen2.5:7B（通过 Ollama） |
| 云端模型 | DeepSeek（OpenAI 兼容 API） |
| RAG 框架 | LangChain |
| API 服务 | FastAPI |
| 业务数据库 | PostgreSQL + SQLAlchemy + Alembic |
| 数据源 | Hugging Face Datasets |

## 配置

编辑 `src/config.py`：

```python
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
OLLAMA_MODEL = "qwen2.5:7b"                 # 对话模型
TOP_K = 3                                    # 检索返回的参考案例数
CHUNK_SIZE = 512                             # 文本分块大小
```

如需使用其他 Ollama 模型（如 `qwen2.5:3b`），修改 `OLLAMA_MODEL` 配置项即可。

模型和数据库建议通过环境变量配置：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DATABASE_URL=postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent
```

## Agent 流程

1. **Question Analyzer Tool**：判断用户问题属于法律问答、类案检索、量刑参考或非法律问题。
2. **Retriever Tool**：根据问题类型选择检索数量，调用 Chroma 返回相关案例。
3. **Answer Generator Tool**：将案例上下文注入提示词，通过 Ollama 生成法律分析。
4. **Citation Checker Tool**：检查回答是否带有参考依据；证据不足时提示无法可靠回答。

## 多轮对话与记忆

系统现在支持 PostgreSQL 持久化：

- `cases` 保存案卷。
- `conversations` 保存案卷下的多轮对话。
- `messages` 保存每轮 user/assistant 消息。
- `case_memories` 保存案件事实摘要、用户目标、争议焦点、已确认结论和待补证点。

`POST /chat` 不传 `conversation_id` 时会自动创建新会话；传入已有 `conversation_id` 时会读取最近 6 轮历史，并注入案件记忆后再生成回答。

## 测试

```bash
pytest -q
```

当前测试覆盖：

- 问题意图识别与检索策略选择
- 非法律问题拒答
- 结构化回答与引用来源
- 检索证据不足时的保守回答
- `/chat`、`/retrieve`、`/health` API 响应结构

## 简历描述参考

- 构建中文法律知识库问答 Agent，基于 Hugging Face 法律案例数据集、Chroma 向量数据库和 bge-small-zh embedding 实现语义检索。
- 设计问题分析、案例检索、答案生成、引用校验等 Agent 工具链，使系统从单轮 RAG 问答升级为可规划、可追溯的法律辅助智能体。
- 使用 FastAPI 封装问答与检索接口，支持结构化返回参考案例、相关法条、罪名信息和风险提示，提升系统可演示性与工程化程度。
- 构建小型评测集验证法律问答、类案检索、无关问题拒答等场景，降低大模型幻觉并提升回答可信度。
