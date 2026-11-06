"""
NutriNexa Chunking Configuration Experiment Script.
Mengevaluasi 3 konfigurasi chunking empiris terhadap seluruh 16 dokumen korpus resmi:
1. Config A: 150 kata / 30 kata overlap
2. Config B: 200 kata / 35 kata overlap
3. Config C: 250 kata / 40 kata overlap

Menggunakan tokenizer riil intfloat/multilingual-e5-small, mengukur metrik teks
(jumlah chunk, distribusi token, deteksi duplikasi/kosong, preservasi batas kalimat),
serta performa temu-balik riil (Hit@1, Hit@3, Hit@5, MRR, Latensi) pada 30 kueri evaluasi.
Mengekspor laporan ke format CSV dan JSON.

Cara Eksekusi:
    uv run python scripts/experiment_chunking.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Pastikan direktori root ada dalam PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from src.knowledge_base.chunker import WordAwareChunker
from src.knowledge_base.document_loader import DocumentLoader
from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.retriever import SemanticRetriever
from src.knowledge_base.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunking_experiment")


CONFIGURATIONS = [
    {"name": "Config_A (150w/30ov)", "chunk_size": 150, "chunk_overlap": 30},
    {"name": "Config_B (200w/35ov)", "chunk_size": 200, "chunk_overlap": 35},
    {"name": "Config_C (250w/40ov)", "chunk_size": 250, "chunk_overlap": 40},
]


def run_experiment() -> None:
    logger.info("=" * 75)
    logger.info("NUTRINEXA T03 — EMPIRICAL CHUNKING CONFIGURATION EXPERIMENT")
    logger.info("=" * 75)

    manifest_path = Path("data/knowledge_base/manifest.json")
    queries_path = Path("data/evaluation/benchmark_queries.json")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists() or not queries_path.exists():
        logger.error("Berkas manifest atau kueri benchmark tidak ditemukan!")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    dev_queries = benchmark_data.get("development_tuning_queries", [])
    if not dev_queries:
        # Fallback kompatibilitas jika format split lama
        dev_queries = benchmark_data.get("queries", [])[:15]

    in_domain_dev = [q for q in dev_queries if q.get("category") != "out_of_distribution"]

    logger.info("Total kueri pengembangan (development split) untuk tuning: %d", len(dev_queries))

    logger.info("Memuat model embedding riil intfloat/multilingual-e5-small...")
    engine = EmbeddingEngine(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=8,
    )
    engine.load_model()
    loader = DocumentLoader()

    # Muat seluruh dokumen korpus
    loaded_docs = []
    for entry in manifest_data["documents"]:
        doc_path = Path(entry["source_path"])
        if doc_path.exists():
            loaded_docs.append(loader.load_file(doc_path, source_id=entry["source_id"], base_metadata=entry))

    logger.info("Total dokumen terverifikasi yang dimuat untuk eksperimen: %d", len(loaded_docs))

    experiment_results: list[dict[str, Any]] = []

    for cfg in CONFIGURATIONS:
        cfg_name = cfg["name"]
        c_size = cfg["chunk_size"]
        c_overlap = cfg["chunk_overlap"]
        logger.info("\n" + "-" * 70)
        logger.info("MENGUJI KONFIGURASI: %s (Size: %d kata, Overlap: %d kata)", cfg_name, c_size, c_overlap)
        logger.info("-" * 70)

        chunker = WordAwareChunker(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            token_counter=engine.count_tokens,
        )

        all_chunks = []
        for doc in loaded_docs:
            all_chunks.extend(chunker.chunk_document(doc))

        total_chunks = len(all_chunks)
        logger.info("  1. Total chunks dihasilkan: %d", total_chunks)

        # Pemeriksaan integritas chunks
        empty_chunks = sum(1 for c in all_chunks if not c.text.strip())
        chunk_ids = [c.chunk_id for c in all_chunks]
        duplicate_ids = len(chunk_ids) - len(set(chunk_ids))

        # Analisis distribusi token E5 riil
        token_counts = [engine.count_tokens(c.text, is_query=False) for c in all_chunks]
        token_arr = np.array(token_counts)
        tok_min = int(np.min(token_arr)) if len(token_arr) > 0 else 0
        tok_max = int(np.max(token_arr)) if len(token_arr) > 0 else 0
        tok_mean = float(np.mean(token_arr)) if len(token_arr) > 0 else 0.0
        tok_median = float(np.median(token_arr)) if len(token_arr) > 0 else 0.0
        tok_p95 = float(np.percentile(token_arr, 95)) if len(token_arr) > 0 else 0.0

        # Boundary preservation: persentase chunk yang berakhir dengan tanda baca kalimat (. ! ? :)
        sentence_endings = sum(1 for c in all_chunks if c.text.rstrip()[-1:] in {".", "!", "?", ":"})
        boundary_preservation_rate = (sentence_endings / total_chunks * 100.0) if total_chunks > 0 else 0.0

        logger.info(
            "  2. Distribusi Token E5: Min=%d, Mean=%.1f, Median=%.1f, P95=%.1f, Max=%d (Limit 512)",
            tok_min,
            tok_mean,
            tok_median,
            tok_p95,
            tok_max,
        )
        logger.info("  3. Empty chunks: %d, Duplicate IDs: %d", empty_chunks, duplicate_ids)
        logger.info("  4. Preservasi Batas Kalimat: %.1f%%", boundary_preservation_rate)

        # Indeksasi ke koleksi ChromaDB sementara untuk pengujian temu-balik
        temp_db_dir = Path(f"storage/temp_chroma_exp_{c_size}_{c_overlap}")
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir, ignore_errors=True)

        temp_store = ChromaVectorStore(
            persist_directory=temp_db_dir,
            collection_name=f"exp_{c_size}_{c_overlap}",
        )

        emb_results = engine.embed_chunks(all_chunks)
        temp_store.add_chunks(all_chunks, [res.embedding for res in emb_results])

        temp_retriever = SemanticRetriever(
            embedding_engine=engine,
            vector_store=temp_store,
            default_top_k=5,
        )

        # Eksekusi kueri evaluasi
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        reciprocal_ranks = []
        latencies = []

        for q in dev_queries:
            q_text = q["query"]
            expected_sources = q.get("expected_source_ids", [])
            is_ood = q.get("category") == "out_of_distribution"

            t0 = time.perf_counter()
            retrieved = temp_retriever.retrieve(q_text, top_k=5)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)

            if not is_ood:
                match_rank = None
                for rank_idx, r in enumerate(retrieved, start=1):
                    if r.metadata.get("source_id") in expected_sources:
                        match_rank = rank_idx
                        break

                if match_rank is not None:
                    if match_rank <= 1:
                        hit_1 += 1
                    if match_rank <= 3:
                        hit_3 += 1
                    if match_rank <= 5:
                        hit_5 += 1
                    reciprocal_ranks.append(1.0 / match_rank)
                else:
                    reciprocal_ranks.append(0.0)

        num_id = len(in_domain_dev)
        hit_1_pct = (hit_1 / num_id * 100.0) if num_id > 0 else 0.0
        hit_3_pct = (hit_3 / num_id * 100.0) if num_id > 0 else 0.0
        hit_5_pct = (hit_5 / num_id * 100.0) if num_id > 0 else 0.0
        mrr_val = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        logger.info(
            "  5. Performa Temu-Balik: Hit@1=%.1f%%, Hit@3=%.1f%%, Hit@5=%.1f%%, MRR=%.4f, Mean Latency=%.2f ms",
            hit_1_pct,
            hit_3_pct,
            hit_5_pct,
            mrr_val,
            mean_lat,
        )

        experiment_results.append({
            "config_name": cfg_name,
            "chunk_size_words": c_size,
            "chunk_overlap_words": c_overlap,
            "total_chunks": total_chunks,
            "empty_chunks": empty_chunks,
            "duplicate_ids": duplicate_ids,
            "token_min": tok_min,
            "token_mean": round(tok_mean, 1),
            "token_median": round(tok_median, 1),
            "token_p95": round(tok_p95, 1),
            "token_max": tok_max,
            "boundary_preservation_pct": round(boundary_preservation_rate, 1),
            "hit_at_1_pct": round(hit_1_pct, 1),
            "hit_at_3_pct": round(hit_3_pct, 1),
            "hit_at_5_pct": round(hit_5_pct, 1),
            "mrr": round(mrr_val, 4),
            "latency_mean_ms": round(mean_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
        })

        # Bersihkan direktori Chroma sementara
        shutil.rmtree(temp_db_dir, ignore_errors=True)

    # Ekspor hasil ke CSV
    csv_file = reports_dir / "chunk_configuration_comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiment_results[0].keys()))
        writer.writeheader()
        writer.writerows(experiment_results)

    json_file = reports_dir / "chunk_configuration_comparison.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info("\n" + "=" * 75)
    logger.info("RINGKASAN EKSPERIMEN PERBANDINGAN KONFIGURASI CHUNKING")
    logger.info("=" * 75)
    print(
        f"{'Konfigurasi':<23} | {'Chunks':<6} | {'Avg Tok':<7} | {'Max Tok':<7} | "
        f"{'Hit@1':<7} | {'Hit@3':<7} | {'MRR':<7} | {'Latensi':<8}"
    )
    print("-" * 75)
    for res in experiment_results:
        print(
            f"{res['config_name']:<23} | {res['total_chunks']:<6} | {res['token_mean']:<7.1f} | "
            f"{res['token_max']:<7} | {res['hit_at_1_pct']:<6.1f}% | {res['hit_at_3_pct']:<6.1f}% | "
            f"{res['mrr']:<7.4f} | {res['latency_mean_ms']:<6.2f}ms"
        )
    print("=" * 75)
    logger.info("Laporan perbandingan berhasil disimpan ke: %s", csv_file)


if __name__ == "__main__":
    run_experiment()

