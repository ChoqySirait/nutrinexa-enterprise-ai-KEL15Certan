import chromadb
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "nutrinexa_kb"):
        """
        Menggunakan PersistentClient agar indeks memori semantik tersimpan permanen di disk lokal.
        Rubrik Milestone 3: Penyimpanan persisten dan metadata tersimpan.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Inisialisasi koleksi dengan konfigurasi hnsw:space = cosine (Cosine Similarity)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )

    def add_documents(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Menyimpan potongan dokumen, vektor, dan metadata ke ChromaDB."""
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Berhasil mengindeks {len(ids)} potongan (chunks) dokumen ke ChromaDB.")

    def search(self, query_embedding: List[float], n_results: int = 3):
        """Mencari dokumen paling relevan berdasarkan kedekatan vektor semantik."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )