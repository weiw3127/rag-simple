"""RAG baseline library."""

__all__ = [
    "load_txt_documents",
    "BM25Retriever",
    "DenseRetriever",
    "rrf_fuse",
    "__version__",
]

__version__ = "1.0.1"

from .ingest import load_txt_documents
from .bm25 import BM25Retriever

# still working 
try: 
    from .dense import DenseRetriever
except Exception: 
    DesnseRetriever = None

try: 
    from .hybird import rrf_fuse
except Exception :
    rrf_fuse = None
