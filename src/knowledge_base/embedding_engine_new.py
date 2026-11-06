from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inisialisasi model embedding untuk bahasa Indonesia (Multilingual).
        Model ini akan mengubah teks dokumen menjadi dense vector.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> List[float]:
        """Menghasilkan vektor untuk satu teks (digunakan saat kueri pencarian)."""
        return self.model.encode(text).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Menghasilkan batch vektor untuk banyak teks (digunakan saat ingestion)."""
        return self.model.encode(texts).tolist()