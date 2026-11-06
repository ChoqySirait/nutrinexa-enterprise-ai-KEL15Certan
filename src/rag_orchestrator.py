import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Memuat variabel environment (API Key Gemini)
load_dotenv()

class NutriNexaAgent:
    def __init__(self):
        print("Menginisialisasi NutriNexa Agentic RAG...")
        # 1. Hubungkan ke Vector Database Lokal (ChromaDB)
        db_path = os.path.join("data", "chroma_db")
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="nutrinexa_kb")
        
        # 2. Siapkan Model Embedding untuk pencarian
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 3. Konfigurasi Google Gemini API dengan pustaka baru
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan. Pastikan sudah ada di file .env")
        self.ai_client = genai.Client(api_key=api_key)

    def ask(self, query: str):
        print(f"\n======================================")
        print(f"User Kueri: {query}")
        
        # Langkah A: Mengubah teks pertanyaan menjadi vektor
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Langkah B: Mencari dokumen SOP gizi di ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=2 
        )
        
        # Langkah C: Menyusun konteks dari hasil pencarian
        context = "\n".join(results['documents'][0]) if results['documents'] else "Tidak ada referensi."
        print(f"\n[Referensi Ditemukan di Database]:\n{context}\n")
        
        # Langkah D: Mengirim Prompt utuh ke Gemini AI
        prompt = f"""
        Anda adalah asisten AI 'NutriNexa'. Tugas Anda adalah memberikan saran gizi dan kuliner.
        Gunakan REFERENSI SOP di bawah ini untuk menjawab pertanyaan pengguna. Jika jawabannya ada di referensi, gunakan itu sebagai acuan utama.
        
        REFERENSI SOP:
        {context}
        
        PERTANYAAN PENGGUNA: 
        {query}
        """
        
        # Menggunakan model standar terbaru (gemini-3.8-flash)
        response = self.ai_client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt
        )
        print(f"[Jawaban NutriNexa]:\n{response.text}")
        print(f"======================================\n")
        return response.text

if __name__ == "__main__":
    agent = NutriNexaAgent()
    
    agent.ask("Berapa batas maksimal garam harian untuk pasien hipertensi?")
    agent.ask("Bagaimana cara memanfaatkan sisa batang pokchoy secara zero-waste?")