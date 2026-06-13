import json
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, RAW_DATA_DIR
from db import create_session_factory, init_db, session_scope
from models import Base, LegalDocument


def load_data(filepath, limit=None):
    docs = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = item.get("meta", {})
            fact = item.get("fact", "")
            accusations = ", ".join(meta.get("accusation", []))
            articles = meta.get("relevant_articles", [])
            punishment = meta.get("term_of_imprisonment", {})

            text = f"案情：{fact}"
            if accusations:
                text += f"\n罪名：{accusations}"
            if articles:
                text += f"\n相关法条：第{'、第'.join(str(a) for a in articles)}条"
            if punishment:
                prison = punishment.get("imprisonment", 0)
                if prison:
                    text += f"\n刑期：{prison}年"

            docs.append(Document(
                page_content=text,
                metadata={
                    "accusations": accusations,
                    "articles": str(articles),
                    "punishment": punishment.get("imprisonment", 0),
                }
            ))
    return docs


def ingest():
    session_factory = create_session_factory()
    init_db(Base.metadata, session_factory)

    with session_scope(session_factory) as session:
        count = session.execute(text("SELECT COUNT(*) FROM legal_documents")).scalar()
        if count and count > 0:
            print(f"PostgreSQL 中已有 {count} 条法律文档，跳过导入")
            return

    train_path = Path(RAW_DATA_DIR) / "train.json"
    if not train_path.exists():
        print("Data not found, run download.py first")
        return

    import os
    limit_env = os.environ.get("INGEST_LIMIT")
    limit = int(limit_env) if limit_env else None
    if limit:
        print(f"Loading data (limit={limit} records)...")
    else:
        print("Loading all data...")
    docs = load_data(str(train_path), limit=limit)
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    del docs
    print(f"Split into {len(chunks)} chunks")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]
    del chunks

    BATCH = 128
    total = len(texts)
    print(f"Starting embedding loop ({total} texts, batch={BATCH})...", flush=True)

    session_factory = create_session_factory()
    with session_scope(session_factory) as session:
        for i in range(0, total, BATCH):
            batch_texts = texts[i:i + BATCH]
            batch_metas = metas[i:i + BATCH]
            t0 = time.time()
            embeddings = embed_model.encode(batch_texts, show_progress_bar=False).tolist()
            t1 = time.time()

            for chunk_text, meta, emb in zip(batch_texts, batch_metas, embeddings):
                session.add(LegalDocument(
                    content=chunk_text,
                    accusations=meta.get("accusations", ""),
                    articles=meta.get("articles", ""),
                    punishment=meta.get("punishment", 0),
                    embedding=emb,
                ))
            session.flush()
            t2 = time.time()
            print(f"  [{min(i + BATCH, total)}/{total}] encode:{t1 - t0:.1f}s db:{t2 - t1:.1f}s")

    print(f"Ingested {total} documents into PostgreSQL")


if __name__ == "__main__":
    ingest()
