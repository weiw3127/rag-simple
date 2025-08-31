from pathlib import Path
from typing import List, Dict

def load_txt_documents(root: str) -> list[dict]:
    docs: List[Dict] = []
    for p in sorted(Path(root).rglob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        docs.append({"id": str(p), "text": txt})
    return docs
