from typing import List, Dict

def build_prompt(question: str, contexts: List[str]) -> str:
    header = "Answer using ONLY the context. Cite as [#]. If unsure, say 'I don't know.'\n\n"
    ctx = "\n\n".join([f"[{i+1}] " + c for i, c in enumerate(contexts)])
    return header + ctx + f"\n\nQuestion: {question}\nAnswer:"

def stub_answer(question: str, contexts: List[Dict]) -> dict:
    texts = [c["text"] for c in contexts]
    answer = texts[0] if texts else "I don't know."
    return {"answer": answer, "citations": [c["id"] for c in contexts][:len(texts)]}
