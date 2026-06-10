import json, time
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, RAW_DATA_DIR

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
    persist = Path(CHROMA_PERSIST_DIR)
    if persist.exists() and any(persist.iterdir()):
        print("Chroma index already exists, skipping ingestion")
        return

    train_path = Path(RAW_DATA_DIR) / "train.json"
    if not train_path.exists():
        print("Data not found, run download.py first")
        return

    print("Loading data (first 1000 records)...")
    docs = load_data(str(train_path), limit=1000)
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
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]
    del chunks

    BATCH = 128
    client = chromadb.PersistentClient(path=str(persist))
    collection = client.get_or_create_collection(name="legal_docs")

    total = len(texts)
    print(f"Starting embedding loop ({total} texts, batch={BATCH})...", flush=True)
    for i in range(0, total, BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_metas = metas[i:i+BATCH]
        t0 = time.time()
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        t1 = time.time()
        ids = [f"doc_{i+j}" for j in range(len(batch_texts))]
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
            ids=ids,
        )
        t2 = time.time()
        print(f"  [{min(i+BATCH, total)}/{total}] encode:{t1-t0:.1f}s db:{t2-t1:.1f}s")

    print(f"Index saved to {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    ingest()
