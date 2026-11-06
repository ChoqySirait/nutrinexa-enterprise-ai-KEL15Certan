import os
import re
import sys
import fitz  # PyMuPDF

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Mengimpor dari file _new yang kamu buat
from knowledge_base.embedding_engine_new import EmbeddingEngine
from knowledge_base.vector_store_new import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_corpus(corpus_dir: str):
    print("\n--- MEMULAI PIPELINE INGESTION NUTRINEXA ---")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    metadatas = []
    ids = []
    chunk_id_counter = 0

    print(f"Membaca dokumen dari: {corpus_dir}")
    pdf_files = [f for f in os.listdir(corpus_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("PERINGATAN: Tidak ada file PDF di folder data/corpus/!")
        return

    for filename in pdf_files:
        filepath = os.path.join(corpus_dir, filename)
        doc = fitz.open(filepath)
        
        full_text = ""
        for page_num in range(len(doc)):
            full_text += doc.load_page(page_num).get_text("text") + " "
        
        cleaned_text = clean_text(full_text)
        doc_chunks = text_splitter.split_text(cleaned_text)
        
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadatas.append({
                "source_document": filename,
                "chunk_index": i
            })
            ids.append(f"doc_{chunk_id_counter}")
            chunk_id_counter += 1

    print(f"Total potongan dokumen (chunks) terekstraksi: {len(chunks)}")

    embedder = EmbeddingEngine()
    vector_db = VectorStore(persist_directory="./chroma_db", collection_name="nutrinexa_kb")
    
    print("Mengekstraksi Dense Vector Embeddings... (mohon tunggu)")
    embeddings = embedder.get_embeddings(chunks)
    
    vector_db.add_documents(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)
    print("--- PIPELINE SELESAI. CHROMADB PERSISTEN TELAH DIPERBARUI ---")

if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus')
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
        print(f"Folder {corpus_path} baru saja dibuat. Masukkan 10-15 PDF ke dalamnya lalu jalankan ulang.")
    else:
        process_corpus(corpus_path)