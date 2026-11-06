"""
NutriNexa Knowledge Base Package (Milestone T03).
Menyediakan modul abstraksi skema, pembersihan dokumen, chunking adaptif,
dense embedding asimetris, penyimpanan vektor persisten ChromaDB, dan semantic retrieval.
"""

from src.knowledge_base.chunker import WordAwareChunker, compare_chunking_configurations
from src.knowledge_base.document_loader import (
    DocumentLoader,
    LoadedDocument,
    PageContent,
)
from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.preprocessing import (
    clean_text,
    extract_frontmatter,
    is_valid_document_text,
)
from src.knowledge_base.retriever import SemanticRetriever
from src.knowledge_base.schemas import (
    DocumentChunk,
    EmbeddingEngineError,
    EmbeddingResult,
    KnowledgeBaseError,
    RetrievalResult,
    RetrieverError,
    SchemaValidationError,
    TokenLimitExceededError,
    VectorStoreError,
)
from src.knowledge_base.vector_store import ChromaVectorStore

__all__ = [
    "ChromaVectorStore",
    "DocumentChunk",
    "DocumentLoader",
    "EmbeddingEngine",
    "EmbeddingEngineError",
    "EmbeddingResult",
    "KnowledgeBaseError",
    "LoadedDocument",
    "PageContent",
    "RetrievalResult",
    "RetrieverError",
    "SchemaValidationError",
    "SemanticRetriever",
    "TokenLimitExceededError",
    "VectorStoreError",
    "WordAwareChunker",
    "clean_text",
    "compare_chunking_configurations",
    "extract_frontmatter",
    "is_valid_document_text",
]
