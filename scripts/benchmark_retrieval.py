import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Mengimpor dari file _new yang kamu buat
from knowledge_base.embedding_engine_new import EmbeddingEngine
from knowledge_base.vector_store_new import VectorStore

def run_benchmark():
    embedder = EmbeddingEngine()
    vector_db = VectorStore(persist_directory="./chroma_db", collection_name="nutrinexa_kb")

    test_queries = [
        "Bagaimana standar gizi dan kalori harian untuk mahasiswa di asrama?",  
        "Aturan pengurangan sisa makanan (zero-waste) di kantin",               
        "Panduan menu nabati bagi mahasiswa yang memiliki alergi",              
        "Prosedur pengambilan porsi makanan khusus jika sedang sakit",          
        "Bagaimana cara reset password lisensi Microsoft 365?"                  
    ]

    DISTANCE_THRESHOLD = 0.60 

    results_data = []

    print("\n" + "="*60)
    print(" BENCHMARK TEMU BALIK SEMANTIK NUTRINEXA (MILESTONE 3)")
    print("="*60)

    for q in test_queries:
        print(f"\nKueri: '{q}'")
        q_emb = embedder.get_embedding(q)
        
        search_res = vector_db.search(query_embedding=q_emb, n_results=1)
        
        distances = search_res['distances'][0]
        documents = search_res['documents'][0]
        metadatas = search_res['metadatas'][0]
        
        best_distance = distances[0] if distances else 1.0
        cosine_similarity = 1 - best_distance
        
        if best_distance > DISTANCE_THRESHOLD:
            status = "DITOLAK (Out-of-domain)"
            best_doc = "N/A (Kemiripan terlalu rendah)"
            source = "-"
        else:
            status = "DITERIMA"
            best_doc = documents[0][:150].replace('\n', ' ') + "..."
            source = metadatas[0]['source_document']
            
        print(f"Status           : {status}")
        print(f"Cosine Similarity: {cosine_similarity:.4f}")
        print(f"Dokumen Teratas  : {best_doc}")

        results_data.append({
            "Kueri": q,
            "Skor Cosine": f"{cosine_similarity:.4f}",
            "Status": status,
            "Dokumen Sumber": source
        })

    df = pd.DataFrame(results_data)
    print("\n\n" + "="*60)
    print(" TABEL HASIL BENCHMARK (Salin ke Dokumen Laporan Anda)")
    print("="*60)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()