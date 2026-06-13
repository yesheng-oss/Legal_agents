import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from download import download
from ingest import ingest
from rag import LegalRAG

def main():
    print("=" * 50)
    print("  法律 RAG 知识库系统")
    print("=" * 50)

    if not (Path("data/chroma").exists() and any(Path("data/chroma").iterdir())):
        if input("\n1. 下载数据 (y/n): ").lower() in ("y", "yes"):
            download()
        if input("2. 构建索引 (y/n): ").lower() in ("y", "yes"):
            ingest()

    print("\n3. 启动查询")
    rag = LegalRAG()

    while True:
        q = input("\n问题 (输入 quit 退出): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer, refs = rag.query(q)
        print(f"\n答案: {answer}")
        print(f"\n参考来源: {len(refs)} 条案例")

if __name__ == "__main__":
    main()
