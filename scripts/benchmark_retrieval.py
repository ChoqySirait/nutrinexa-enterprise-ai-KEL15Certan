"""
NutriNexa Comprehensive Semantic Retrieval Benchmark Suite (Audit-Corrected).
Mengimplementasikan metrik temu-balik mathematically sound:
1. Precision@K (Chunk-level & Document-level terdefinisi ketat <= 1.0).
2. Recall@K (Berdasarkan irisan himpunan sumber unik relevan, strictly <= 1.0).
3. Hit@K & Mean Reciprocal Rank (MRR).
4. Kalibrasi Ambang Batas OOD terpisah tanpa kebocoran (disjoint calibration vs held-out splits).
5. Klasifikasi Keputusan OOD Eksplisit (True Rejection, False Acceptance, Abstain Band).
6. Profil Latensi Terstandarisasi: Model load, Warmup pass, Warm query (mean, median, p95).

Cara Eksekusi:
    uv run python scripts/benchmark_retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Pastikan direktori root proyek ada dalam PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from src.knowledge_base.embedding_engine import EmbeddingEngine
from src.knowledge_base.retriever import SemanticRetriever
from src.knowledge_base.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NutriNexa Rigorous Semantic Retrieval Benchmark")
    parser.add_argument(
        "--queries",
        type=str,
        default="data/evaluation/benchmark_queries.json",
        help="Jalur berkas kueri evaluasi benchmark (default: data/evaluation/benchmark_queries.json)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=ChromaVectorStore.DEFAULT_PERSIST_DIR,
        help="Direktori ChromaDB yang akan diuji (default: storage/chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=ChromaVectorStore.DEFAULT_COLLECTION_NAME,
        help="Nama koleksi ChromaDB (default: nutrinexa_knowledge_base)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Jumlah dokumen teratas yang dievaluasi per kueri (default: 5)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Direktori penyimpanan laporan keluaran (default: reports)",
    )
    return parser.parse_args()


# ==============================================================================
# FUNGSI METRIK EVALUASI TERSTANDARISASI & TERUJI (MATHEMATICALLY SOUND)
# ==============================================================================


def calculate_hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Hit@K: 1.0 jika setidaknya satu sumber relevan muncul dalam top-K, 0.0 jika tidak."""
    if not expected_sources or k <= 0:
        return 0.0
    top_k_sources = set(retrieved_sources[:k])
    return 1.0 if bool(top_k_sources & set(expected_sources)) else 0.0


def calculate_mrr(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Mean Reciprocal Rank: 1 / rank kemunculan sumber relevan pertama kali dalam top-K."""
    if not expected_sources or k <= 0:
        return 0.0
    expected_set = set(expected_sources)
    for rank_idx, src in enumerate(retrieved_sources[:k], start=1):
        if src in expected_set:
            return 1.0 / rank_idx
    return 0.0


def calculate_document_recall_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int
) -> float:
    """
    Document-level Recall@K:
    Fraksi sumber unik relevan yang berhasil ditemukan dalam top-K terhadap total sumber yang diharapkan.
    Recall@K = |set(retrieved[:K]) & set(expected)| / |set(expected)|
    Dijamin matematis berada dalam rentang [0.0, 1.0] (0% s.d. 100%),
    tidak dapat terinflasi oleh multiple chunks dari source_id yang sama.
    """
    if not expected_sources or k <= 0:
        return 0.0
    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources[:k])
    intersect = retrieved_set & expected_set
    return len(intersect) / len(expected_set)


def calculate_chunk_precision_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int
) -> float:
    """
    Chunk-level Precision@K:
    Fraksi dari K chunk yang ditarik yang berasal dari sumber relevan.
    Precision@K = sum(I(src in expected)) / K
    Dijamin berada dalam rentang [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    if not expected_sources:
        return 0.0
    expected_set = set(expected_sources)
    relevant_chunks = sum(1 for src in retrieved_sources[:k] if src in expected_set)
    return relevant_chunks / k


def calculate_document_precision_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int
) -> float:
    """
    Document-level Precision@K:
    Jumlah sumber unik relevan dalam top-K dibagi K.
    Precision@K_doc = |set(retrieved[:K]) & set(expected)| / K
    Dijamin berada dalam rentang [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    if not expected_sources:
        return 0.0
    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources[:k])
    return len(retrieved_set & expected_set) / k


# ==============================================================================
# PIPELINE BENCHMARK
# ==============================================================================


def calibrate_threshold_split(
    retriever: SemanticRetriever,
    calibration_queries: list[dict[str, Any]],
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Melakukan kalibrasi ambang batas OOD kemiripan secara ketat
    hanya pada kueri kalibrasi (calibration split) tanpa kebocoran data.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]

    id_queries = [q for q in calibration_queries if q.get("category") != "out_of_distribution"]
    ood_queries = [q for q in calibration_queries if q.get("category") == "out_of_distribution"]

    id_similarities: list[float] = []
    for q in id_queries:
        res = retriever.retrieve(q["query"], top_k=1)
        id_similarities.append(res[0].similarity if res else 0.0)

    ood_similarities: list[float] = []
    for q in ood_queries:
        res = retriever.retrieve(q["query"], top_k=1)
        ood_similarities.append(res[0].similarity if res else 0.0)

    calibration_records = []
    best_threshold = 0.85
    min_error = float("inf")

    for th in candidate_thresholds:
        # False Acceptance: OOD query tapi similarity >= th
        fa_count = sum(1 for sim in ood_similarities if sim >= th)
        far = (fa_count / len(ood_similarities)) if ood_similarities else 0.0

        # False Rejection: In-domain query tapi similarity < th
        fr_count = sum(1 for sim in id_similarities if sim < th)
        frr = (fr_count / len(id_similarities)) if id_similarities else 0.0

        total_err = far + frr
        calibration_records.append({
            "threshold": th,
            "false_acceptance_rate": round(far, 4),
            "false_rejection_rate": round(frr, 4),
            "total_error": round(total_err, 4),
        })

        if total_err < min_error:
            min_error = total_err
            best_threshold = th

    return best_threshold, calibration_records


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    queries_file = Path(args.queries)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not queries_file.exists():
        logger.error("Berkas kueri benchmark '%s' tidak ditemukan!", queries_file)
        sys.exit(1)

    with open(queries_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    calib_queries = bench_data.get("calibration_queries", [])
    eval_queries = bench_data.get("held_out_evaluation_queries", [])
    if not eval_queries:
        eval_queries = bench_data.get("evaluation_queries", [])

    logger.info("=" * 75)
    logger.info("NUTRINEXA T03 — RIGOROUS RETRIEVAL BENCHMARK & METRICS AUDIT")
    logger.info("=" * 75)
    logger.info("Kueri Kalibrasi : %d entri", len(calib_queries))
    logger.info("Kueri Evaluasi  : %d entri (Held-Out Final Split)", len(eval_queries))
    logger.info("ChromaDB Path   : %s", args.db_dir)
    logger.info("Koleksi Target  : %s", args.collection)

    # 1. Pengukuran Model Loading Latency (True Cold-Start Component)
    t_load_start = time.perf_counter()
    engine = EmbeddingEngine(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=8,
    )
    engine.load_model()
    t_load_end = time.perf_counter()
    model_loading_latency_ms = (t_load_end - t_load_start) * 1000.0

    store = ChromaVectorStore(
        persist_directory=args.db_dir,
        collection_name=args.collection,
    )

    doc_count = store.count()
    logger.info("Total Chunks Tersedia di Database: %d", doc_count)
    if doc_count == 0:
        logger.error("Database ChromaDB kosong! Jalankan scripts/ingest_corpus.py terlebih dahulu.")
        sys.exit(1)

    retriever = SemanticRetriever(
        embedding_engine=engine,
        vector_store=store,
        default_top_k=args.top_k,
    )

    # 2. Kalibrasi Ambang Batas pada Dataset Kalibrasi
    logger.info("\n[FASE 1] Menjalankan Kalibrasi Ambang Batas OOD pada %d kueri kalibrasi...", len(calib_queries))
    optimal_threshold, calib_records = calibrate_threshold_split(retriever, calib_queries)
    logger.info("Ambang Batas Terkalibrasi Optimal (Frozen): %.2f", optimal_threshold)

    # 3. Evaluasi Kueri Uji Final Held-Out (35 Kueri: 23 ID + 12 OOD)
    logger.info("\n[FASE 2] Menjalankan Evaluasi Temu-Balik Final pada %d kueri held-out...", len(eval_queries))
    in_domain_eval = [
        q for q in eval_queries
        if q.get("query_type") != "out_of_distribution" and q.get("category") != "out_of_distribution"
    ]
    ood_eval = [
        q for q in eval_queries
        if q.get("query_type") == "out_of_distribution" or q.get("category") == "out_of_distribution"
    ]

    hit_at_1_list: list[float] = []
    hit_at_3_list: list[float] = []
    hit_at_5_list: list[float] = []
    chunk_precision_list: list[float] = []
    doc_precision_list: list[float] = []
    doc_recall_at_1_list: list[float] = []
    doc_recall_at_5_list: list[float] = []
    mrr_list: list[float] = []

    warm_latencies_ms: list[float] = []
    first_query_latency_ms: float = 0.0

    per_query_records: list[dict[str, Any]] = []

    for idx, item in enumerate(eval_queries):
        qid = item.get("query_id", f"EVAL_{idx+1:02d}")
        q_text = item["query"]
        expected_sources = item.get("expected_source_ids", [])
        is_ood = (
            item.get("query_type") == "out_of_distribution"
            or item.get("category") == "out_of_distribution"
            or len(expected_sources) == 0
        )

        # Pengukuran latensi
        t0 = time.perf_counter()
        results = retriever.retrieve(q_text, top_k=args.top_k)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        if idx == 0:
            first_query_latency_ms = latency_ms
        else:
            warm_latencies_ms.append(latency_ms)

        retrieved_sources = [r.metadata.get("source_id", "") for r in results]
        top_similarity = results[0].similarity if results else 0.0

        # Kerangka Keputusan Tiga Arah OOD (Strict Three-Way Decision Framework)
        # REJECT: top_similarity < 0.840
        # ABSTAIN: 0.840 <= top_similarity <= 0.855 inclusive
        # ACCEPT: top_similarity > 0.855
        abstain_lower = optimal_threshold
        abstain_upper = optimal_threshold + 0.015

        if top_similarity < abstain_lower:
            system_decision = "REJECT_AS_OOD"
        elif top_similarity <= abstain_upper:
            system_decision = "BORDERLINE_ABSTAIN"
        else:
            system_decision = "ACCEPT_IN_DOMAIN"

        # Evaluasi Kebenaran Keputusan Tiga Arah (Eksklusif, Tanpa Duplikasi)
        if is_ood:
            if system_decision == "REJECT_AS_OOD":
                decision_outcome = "TRUE_REJECTION"
            elif system_decision == "BORDERLINE_ABSTAIN":
                decision_outcome = "ABSTAIN"
            else:
                decision_outcome = "FALSE_ACCEPTANCE"
        else:
            if system_decision == "ACCEPT_IN_DOMAIN":
                decision_outcome = "TRUE_ACCEPTANCE"
            elif system_decision == "BORDERLINE_ABSTAIN":
                decision_outcome = "ABSTAIN"
            else:
                decision_outcome = "FALSE_REJECTION"

        # Perhitungan Metrik Temu-Balik Kuantitatif untuk Kueri In-Domain
        first_match_rank = 0
        h1 = 0.0
        h3 = 0.0
        h5 = 0.0
        mrr_val = 0.0
        rec_doc_1 = 0.0
        rec_doc_5 = 0.0
        prec_chunk = 0.0
        prec_doc = 0.0

        if not is_ood:
            h1 = calculate_hit_at_k(retrieved_sources, expected_sources, 1)
            h3 = calculate_hit_at_k(retrieved_sources, expected_sources, 3)
            h5 = calculate_hit_at_k(retrieved_sources, expected_sources, 5)
            mrr_val = calculate_mrr(retrieved_sources, expected_sources, args.top_k)
            rec_doc_1 = calculate_document_recall_at_k(retrieved_sources, expected_sources, 1)
            rec_doc_5 = calculate_document_recall_at_k(retrieved_sources, expected_sources, args.top_k)
            prec_chunk = calculate_chunk_precision_at_k(retrieved_sources, expected_sources, args.top_k)
            prec_doc = calculate_document_precision_at_k(retrieved_sources, expected_sources, args.top_k)

            # Hitung first_match_rank untuk tampilan laporan
            expected_set = set(expected_sources)
            for r_idx, src in enumerate(retrieved_sources[: args.top_k], start=1):
                if src in expected_set:
                    first_match_rank = r_idx
                    break

            hit_at_1_list.append(h1)
            hit_at_3_list.append(h3)
            hit_at_5_list.append(h5)
            mrr_list.append(mrr_val)
            doc_recall_at_1_list.append(rec_doc_1)
            doc_recall_at_5_list.append(rec_doc_5)
            chunk_precision_list.append(prec_chunk)
            doc_precision_list.append(prec_doc)

        status_display = decision_outcome if is_ood else (f"HIT_R{first_match_rank}" if first_match_rank > 0 else "MISS")

        per_query_records.append({
            "query_id": qid,
            "query": q_text,
            "category": item.get("category", "general"),
            "intent": item.get("intent", "direct_factual"),
            "expected_sources": "|".join(expected_sources) if expected_sources else "NONE",
            "top_retrieved_source": retrieved_sources[0] if retrieved_sources else "NONE",
            "retrieved_sources": "|".join(retrieved_sources[: args.top_k]),
            "top_similarity": round(top_similarity, 4),
            "first_match_rank": first_match_rank,
            "recall_at_1": round(rec_doc_1, 4),
            "recall_at_5": round(rec_doc_5, 4),
            "chunk_precision_at_5": round(prec_chunk, 4),
            "doc_precision_at_5": round(prec_doc, 4),
            "latency_ms": round(latency_ms, 2),
            "is_ood": is_ood,
            "system_decision": system_decision,
            "decision_outcome": decision_outcome,
            "status": status_display,
        })

    # Perhitungan Statistik Agregat Makro & Mikro
    num_id = len(in_domain_eval)
    num_ood = len(ood_eval)

    macro_hit_1 = float(np.mean(hit_at_1_list)) if hit_at_1_list else 0.0
    macro_hit_3 = float(np.mean(hit_at_3_list)) if hit_at_3_list else 0.0
    macro_hit_5 = float(np.mean(hit_at_5_list)) if hit_at_5_list else 0.0
    macro_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    macro_doc_recall_at_1 = float(np.mean(doc_recall_at_1_list)) if doc_recall_at_1_list else 0.0
    macro_doc_recall_at_5 = float(np.mean(doc_recall_at_5_list)) if doc_recall_at_5_list else 0.0
    macro_chunk_precision = float(np.mean(chunk_precision_list)) if chunk_precision_list else 0.0
    macro_doc_precision = float(np.mean(doc_precision_list)) if doc_precision_list else 0.0

    # Perhitungan Mikro Recall: sum(|set(retrieved[:k]) & set(expected)|) / sum(|set(expected)|)
    total_relevant_found_k5 = sum(
        len(set(r["retrieved_sources"].split("|")[: args.top_k]) & set(r["expected_sources"].split("|")))
        for r in per_query_records
        if not r["is_ood"]
    )
    total_relevant_found_k1 = sum(
        len(set(r["retrieved_sources"].split("|")[:1]) & set(r["expected_sources"].split("|")))
        for r in per_query_records
        if not r["is_ood"]
    )
    total_relevant_possible = sum(len(q.get("expected_source_ids", [])) for q in in_domain_eval)
    micro_doc_recall_at_5 = (total_relevant_found_k5 / total_relevant_possible) if total_relevant_possible > 0 else 0.0
    micro_doc_recall_at_1 = (total_relevant_found_k1 / total_relevant_possible) if total_relevant_possible > 0 else 0.0

    # Analisis Keputusan Tiga Arah OOD
    ood_true_rejections = sum(1 for r in per_query_records if r["is_ood"] and r["decision_outcome"] == "TRUE_REJECTION")
    ood_abstentions = sum(1 for r in per_query_records if r["is_ood"] and r["decision_outcome"] == "ABSTAIN")
    ood_false_acceptances = sum(1 for r in per_query_records if r["is_ood"] and r["decision_outcome"] == "FALSE_ACCEPTANCE")
    ood_trr = (ood_true_rejections / num_ood) if num_ood > 0 else 0.0
    ood_abstention_rate = (ood_abstentions / num_ood) if num_ood > 0 else 0.0
    ood_far = (ood_false_acceptances / num_ood) if num_ood > 0 else 0.0

    id_true_acceptances = sum(1 for r in per_query_records if not r["is_ood"] and r["decision_outcome"] == "TRUE_ACCEPTANCE")
    id_abstentions = sum(1 for r in per_query_records if not r["is_ood"] and r["decision_outcome"] == "ABSTAIN")
    id_false_rejections = sum(1 for r in per_query_records if not r["is_ood"] and r["decision_outcome"] == "FALSE_REJECTION")
    id_tar = (id_true_acceptances / num_id) if num_id > 0 else 0.0
    id_abstention_rate = (id_abstentions / num_id) if num_id > 0 else 0.0
    id_frr = (id_false_rejections / num_id) if num_id > 0 else 0.0

    total_abstentions = ood_abstentions + id_abstentions
    total_abstention_rate = (total_abstentions / len(eval_queries)) if eval_queries else 0.0
    total_decisive_correct = ood_true_rejections + id_true_acceptances
    total_decisive_rate = (total_decisive_correct / len(eval_queries)) if eval_queries else 0.0

    # Statistik Latensi
    warm_lat_arr = np.array(warm_latencies_ms) if warm_latencies_ms else np.array([first_query_latency_ms])
    warm_mean_lat = float(np.mean(warm_lat_arr))
    warm_median_lat = float(np.median(warm_lat_arr))
    warm_min_lat = float(np.min(warm_lat_arr))
    warm_max_lat = float(np.max(warm_lat_arr))
    warm_p95_lat = float(np.percentile(warm_lat_arr, 95))

    total_cold_start_latency_ms = model_loading_latency_ms + first_query_latency_ms

    summary = {
        "corpus_total_chunks": doc_count,
        "calibration_queries_count": len(calib_queries),
        "held_out_evaluation_queries_count": len(eval_queries),
        "in_domain_queries_count": num_id,
        "out_of_distribution_queries_count": num_ood,
        "three_way_threshold_bands": {
            "reject_below": optimal_threshold,
            "abstain_lower": optimal_threshold,
            "abstain_upper": optimal_threshold + 0.015,
            "accept_above": optimal_threshold + 0.015,
        },
        "hit_at_1_macro": round(macro_hit_1, 4),
        "hit_at_3_macro": round(macro_hit_3, 4),
        "hit_at_5_macro": round(macro_hit_5, 4),
        "document_recall_at_1_macro": round(macro_doc_recall_at_1, 4),
        "document_recall_at_1_micro": round(micro_doc_recall_at_1, 4),
        "document_recall_at_5_macro": round(macro_doc_recall_at_5, 4),
        "document_recall_at_5_micro": round(micro_doc_recall_at_5, 4),
        "chunk_precision_at_5_macro": round(macro_chunk_precision, 4),
        "document_precision_at_5_macro": round(macro_doc_precision, 4),
        "mean_reciprocal_rank_mrr_macro": round(macro_mrr, 4),
        "ood_three_way_decision_breakdown": {
            "ood_total_queries": num_ood,
            "ood_true_rejections_count": ood_true_rejections,
            "ood_true_rejection_rate_trr": round(ood_trr, 4),
            "ood_abstentions_count": ood_abstentions,
            "ood_abstention_rate": round(ood_abstention_rate, 4),
            "ood_false_acceptances_count": ood_false_acceptances,
            "ood_false_acceptance_rate_far": round(ood_far, 4),
            "in_domain_total_queries": num_id,
            "in_domain_true_acceptances_count": id_true_acceptances,
            "in_domain_true_acceptance_rate_tar": round(id_tar, 4),
            "in_domain_abstentions_count": id_abstentions,
            "in_domain_abstention_rate": round(id_abstention_rate, 4),
            "in_domain_false_rejections_count": id_false_rejections,
            "in_domain_false_rejection_rate_frr": round(id_frr, 4),
            "total_abstentions_count": total_abstentions,
            "total_abstention_rate": round(total_abstention_rate, 4),
            "total_decisive_accuracy_rate": round(total_decisive_rate, 4),
        },
        "latency_profiles_ms": {
            "model_loading_time_ms": round(model_loading_latency_ms, 2),
            "first_query_warmup_ms": round(first_query_latency_ms, 2),
            "total_cold_start_ms": round(total_cold_start_latency_ms, 2),
            "warm_latency_mean_ms": round(warm_mean_lat, 2),
            "warm_latency_median_p50_ms": round(warm_median_lat, 2),
            "warm_latency_min_ms": round(warm_min_lat, 2),
            "warm_latency_max_ms": round(warm_max_lat, 2),
            "warm_latency_p95_ms": round(warm_p95_lat, 2),
        },
    }

    # Cetak Tabel Evaluasi
    print("\n" + "=" * 95)
    print(
        f"{'QID':<7} | {'Kategori':<23} | {'Simil':<7} | {'Rank':<5} | {'Recall':<7} | "
        f"{'Prec':<7} | {'Latensi':<8} | {'Keputusan OOD':<16}"
    )
    print("-" * 95)
    for r in per_query_records:
        rank_str = str(r["first_match_rank"]) if r["first_match_rank"] > 0 else "-"
        rec_str = f"{r['recall_at_5']*100:.0f}%" if not r["is_ood"] else "-"
        prec_str = f"{r['chunk_precision_at_5']*100:.0f}%" if not r["is_ood"] else "-"
        print(
            f"{r['query_id']:<7} | {r['category']:<23} | {r['top_similarity']:<7.4f} | "
            f"{rank_str:<5} | {rec_str:<7} | {prec_str:<7} | {r['latency_ms']:<6.1f}ms | {r['status']:<16}"
        )
    print("=" * 95)

    print("\n" + "=" * 75)
    print("RINGKASAN METRIK TEMU-BALIK SEMANTIK (MATHEMATICALLY VERIFIED):")
    print("=" * 75)
    print(f"  - Hit@1 (Macro)                   : {summary['hit_at_1_macro'] * 100:.1f}%")
    print(f"  - Hit@3 (Macro)                   : {summary['hit_at_3_macro'] * 100:.1f}%")
    print(f"  - Hit@5 (Macro)                   : {summary['hit_at_5_macro'] * 100:.1f}%")
    print(f"  - Document Recall@1 (Macro)       : {summary['document_recall_at_1_macro'] * 100:.1f}%")
    print(f"  - Document Recall@1 (Micro)       : {summary['document_recall_at_1_micro'] * 100:.1f}%")
    print(f"  - Document Recall@5 (Macro)       : {summary['document_recall_at_5_macro'] * 100:.1f}% (Batas <= 100%)")
    print(f"  - Document Recall@5 (Micro)       : {summary['document_recall_at_5_micro'] * 100:.1f}% (Batas <= 100%)")
    print(f"  - Chunk Precision@5 (Macro)       : {summary['chunk_precision_at_5_macro'] * 100:.1f}%")
    print(f"  - Document Precision@5 (Macro)    : {summary['document_precision_at_5_macro'] * 100:.1f}%")
    print(f"  - Mean Reciprocal Rank (MRR)      : {summary['mean_reciprocal_rank_mrr_macro']:.4f}")
    print(f"\nANALISIS KEPUTUSAN TIGA ARAH OOD (Three-Way Decision: Reject < {optimal_threshold:.3f}, Abstain [{optimal_threshold:.3f}, {optimal_threshold+0.015:.3f}], Accept > {optimal_threshold+0.015:.3f}):")
    print(f"  - OOD True Rejections             : {ood_true_rejections} / {num_ood} (TRR: {ood_trr*100:.1f}%)")
    print(f"  - OOD Abstentions (Borderline)    : {ood_abstentions} / {num_ood} (Abstain Rate: {ood_abstention_rate*100:.1f}%)")
    print(f"  - OOD False Acceptances (Type I)  : {ood_false_acceptances} / {num_ood} (FAR: {ood_far*100:.1f}%)")
    print(f"  - In-Domain True Acceptances      : {id_true_acceptances} / {num_id} (TAR: {id_tar*100:.1f}%)")
    print(f"  - In-Domain Abstentions           : {id_abstentions} / {num_id} (Abstain Rate: {id_abstention_rate*100:.1f}%)")
    print(f"  - In-Domain False Rejections      : {id_false_rejections} / {num_id} (FRR: {id_frr*100:.1f}%)")
    print(f"  - Total Abstentions               : {total_abstentions} / {len(eval_queries)} ({total_abstention_rate*100:.1f}%)")
    print(f"  - Total Decisive Accuracy         : {total_decisive_correct} / {len(eval_queries)} ({total_decisive_rate*100:.1f}%)")
    print("\nPROFIL LATENSI CPU TERUKUR:")
    print(f"  - Model Loading Time              : {summary['latency_profiles_ms']['model_loading_time_ms']:.2f} ms")
    print(f"  - First Query Warmup              : {summary['latency_profiles_ms']['first_query_warmup_ms']:.2f} ms")
    print(f"  - Total Cold-Start                : {summary['latency_profiles_ms']['total_cold_start_ms']:.2f} ms")
    print(f"  - Warm Query Latency (Mean)       : {summary['latency_profiles_ms']['warm_latency_mean_ms']:.2f} ms")
    print(f"  - Warm Query Latency (Median P50) : {summary['latency_profiles_ms']['warm_latency_median_p50_ms']:.2f} ms")
    print(f"  - Warm Query Latency (P95)        : {summary['latency_profiles_ms']['warm_latency_p95_ms']:.2f} ms")
    print("=" * 75)

    # Ekspor laporan
    csv_path = reports_dir / "benchmark_retrieval_raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_records)

    json_path = reports_dir / "retrieval_benchmark.json"
    full_report = {
        "summary": summary,
        "calibration_results": calib_records,
        "per_query_results": per_query_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    summary_json_path = reports_dir / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Laporan evaluasi mentah tersimpan di : %s", csv_path)
    logger.info("Laporan evaluasi JSON tersimpan di   : %s", json_path)
    logger.info("Ringkasan evaluasi JSON tersimpan di : %s", summary_json_path)

    return full_report


if __name__ == "__main__":
    run_benchmark(parse_args())
