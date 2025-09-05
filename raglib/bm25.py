from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re

_token = re.compile(r"\b\w+\b")

def _tok(s: str) -> list[str]:
    return _token.findall(s.lower())

class BM25Retriever:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.corpus_tokens = [_tok(d["text"]) for d in docs]
        self.model = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 10) -> list[Tuple[str, float]]:
        q = _tok(query)
        scores = self.model.get_scores(q)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i]["id"], float(scores[i])) for i in idx]