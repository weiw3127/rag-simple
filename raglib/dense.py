from typing import List, Dict, Tuple
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _OK = True
except Exception:
    _OK = False

class DenseRetriever:
    def __init__(self, docs: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not _OK:
            raise ImportError("Install sentence-transformers and faiss-cpu to use DenseRetriever.")
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode([d["text"] for d in docs], convert_to_numpy=True, normalize_embeddings=True)
        d = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.emb)

    def search(self, query: str, k: int = 10) -> list[Tuple[str, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        return [(self.docs[i]["id"], float(D[0][j])) for j, i in enumerate(I[0])]
