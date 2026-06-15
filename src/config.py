EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512  # bge-small-zh-v1.5 输出维度
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
RAW_DATA_DIR = "data/raw"
TOP_K = 3
OLLAMA_TIMEOUT = 120
OLLAMA_MAX_TOKENS = 1024

# 混合检索与 Rerank 配置
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 10  # Rerank 前召回的候选数
KEYWORD_TOP_K = 10  # 关键词检索返回数
VECTOR_TOP_K = 10   # 向量检索返回数
RRF_K = 60          # Reciprocal Rank Fusion 参数
