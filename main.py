import argparse
from raglib.ingest import load_txt_documents
from raglib.bm25 import BM25Retriever
from raglib.generate import stub_answer

# Optional imports for dense/hybrid
try:
    from raglib.dense import DenseRetriever
    from raglib.hybrid import rrf_fuse
    _HAS_DENSE = True
except Exception:
    _HAS_DENSE = False

def retrieve(query: str, docs, mode: str, topk: int):
    mode = mode.lower()
    if mode == "bm25":
        bm25 = BM25Retriever(docs)
        return bm25.search(query, k=topk)
    elif mode == "dense":
        if not _HAS_DENSE:
            raise RuntimeError("Dense mode needs 'sentence-transformers' and 'faiss-cpu' installed.")
        dense = DenseRetriever(docs)
        return dense.search(query, k=topk)
    elif mode == "hybrid":
        if not _HAS_DENSE:
            raise RuntimeError("Hybrid mode needs 'sentence-transformers' and 'faiss-cpu' installed.")
        bm25 = BM25Retriever(docs)
        dense = DenseRetriever(docs)
        b = bm25.search(query, k=topk)
        d = dense.search(query, k=topk)
        return rrf_fuse([b, d], k=60, top_k=topk)
    else:
        raise ValueError("mode must be one of: bm25 | dense | hybrid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--docs", type=str, default="data/docs")
    ap.add_argument("--mode", type=str, default="bm25", choices=["bm25","dense","hybrid"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    docs = load_txt_documents(args.docs)

    ranked = retrieve(args.question, docs, args.mode, args.topk)

    id2doc = {d["id"]: d for d in docs}
    contexts = [{"id": did, "text": id2doc[did]["text"]} for (did, _s) in ranked]

    result = stub_answer(args.question, contexts)

    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== CITATIONS ===")
    for i, cid in enumerate(result["citations"], 1):
        print(f"[{i}] {cid}")

if __name__ == "__main__":
    main()
