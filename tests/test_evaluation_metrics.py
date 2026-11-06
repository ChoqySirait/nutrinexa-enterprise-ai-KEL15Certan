"""
Unit tests untuk fungsi metrik evaluasi temu-balik, konsistensi Recall Makro vs Mikro,
kerangka keputusan tiga arah OOD (Reject, Abstain, Accept), integritas 10 kategori taksonomi,
dan validasi sumber regulasi BPOM & PGRS.
"""

import json
from pathlib import Path

from scripts.benchmark_retrieval import (
    calculate_chunk_precision_at_k,
    calculate_document_precision_at_k,
    calculate_document_recall_at_k,
    calculate_hit_at_k,
    calculate_mrr,
)


def test_recall_at_k_upper_bound_and_no_inflation():
    """
    Menguji bahwa Recall@K TIDAK PERNAH melebihi 1.0 (100%),
    meskipun seluruh K chunk yang ditarik berasal dari source_id yang sama.
    """
    expected = ["SOP-WASTE-2026-001"]
    retrieved_with_duplicates = [
        "SOP-WASTE-2026-001",
        "SOP-WASTE-2026-001",
        "SOP-WASTE-2026-001",
        "SOP-WASTE-2026-001",
        "SOP-WASTE-2026-001",
    ]

    recall = calculate_document_recall_at_k(retrieved_with_duplicates, expected, k=5)
    assert recall == 1.0
    assert 0.0 <= recall <= 1.0


def test_recall_at_k_partial_and_empty():
    """Menguji perhitungan Recall@K parsial dan penanganan kondisi kosong."""
    expected = ["DOC-A", "DOC-B", "DOC-C"]
    retrieved = ["DOC-A", "DOC-IRRELEVANT", "DOC-B", "DOC-X", "DOC-Y"]

    # Menemukan 2 dari 3 expected
    recall = calculate_document_recall_at_k(retrieved, expected, k=5)
    assert abs(recall - (2.0 / 3.0)) < 1e-5

    # k=1 hanya menemukan DOC-A (1 dari 3)
    recall_k1 = calculate_document_recall_at_k(retrieved, expected, k=1)
    assert abs(recall_k1 - (1.0 / 3.0)) < 1e-5

    # Expected kosong
    assert calculate_document_recall_at_k(retrieved, [], k=5) == 0.0
    # Retrieved kosong
    assert calculate_document_recall_at_k([], expected, k=5) == 0.0


def test_macro_and_micro_recall_consistency():
    """
    Menguji konsistensi perhitungan Macro Recall dan Micro Recall multi-kueri.
    Memastikan perbedaan perilaku saat partial matches dan kesamaan saat perfect matches.
    """
    # Skenario 1: Perfect retrieval (semua expected ditemukan dalam top-5)
    # Query 1: 1 expected, 1 found
    # Query 2: 2 expected, 2 found
    q1_expected = ["DOC-A"]
    q1_retrieved = ["DOC-A", "DOC-X", "DOC-Y"]
    q2_expected = ["DOC-B", "DOC-C"]
    q2_retrieved = ["DOC-B", "DOC-C", "DOC-Z"]

    rec_q1 = calculate_document_recall_at_k(q1_retrieved, q1_expected, k=5)
    rec_q2 = calculate_document_recall_at_k(q2_retrieved, q2_expected, k=5)
    macro_recall_k5 = (rec_q1 + rec_q2) / 2.0
    assert macro_recall_k5 == 1.0

    # Micro recall: total relevant found / total expected across queries = (1 + 2) / (1 + 2) = 1.0
    total_found = len(set(q1_retrieved[:5]) & set(q1_expected)) + len(set(q2_retrieved[:5]) & set(q2_expected))
    total_expected = len(q1_expected) + len(q2_expected)
    micro_recall_k5 = total_found / total_expected
    assert micro_recall_k5 == 1.0

    # Skenario 2: Top-1 retrieval (q1 hit rank 1, q2 hit 1 dari 2 pada rank 1)
    rec_q1_k1 = calculate_document_recall_at_k(q1_retrieved, q1_expected, k=1)  # 1/1 = 1.0
    rec_q2_k1 = calculate_document_recall_at_k(q2_retrieved, q2_expected, k=1)  # 1/2 = 0.5
    macro_recall_k1 = (rec_q1_k1 + rec_q2_k1) / 2.0  # 1.5 / 2 = 0.75
    assert abs(macro_recall_k1 - 0.75) < 1e-5

    total_found_k1 = len(set(q1_retrieved[:1]) & set(q1_expected)) + len(set(q2_retrieved[:1]) & set(q2_expected))
    micro_recall_k1 = total_found_k1 / total_expected  # (1 + 1) / 3 = 2/3 = 0.6667
    assert abs(micro_recall_k1 - (2.0 / 3.0)) < 1e-5


def test_precision_at_k_chunk_and_document():
    """Menguji perhitungan Precision@K level chunk dan level dokumen."""
    expected = ["DOC-A"]
    # 3 chunk relevan dari 5 chunk
    retrieved = ["DOC-A", "DOC-A", "DOC-B", "DOC-A", "DOC-C"]

    chunk_prec = calculate_chunk_precision_at_k(retrieved, expected, k=5)
    assert abs(chunk_prec - 0.6) < 1e-5

    doc_prec = calculate_document_precision_at_k(retrieved, expected, k=5)
    assert abs(doc_prec - 0.2) < 1e-5


def test_hit_at_k_and_mrr():
    """Menguji keabsahan metrik Hit@K dan Mean Reciprocal Rank (MRR)."""
    expected = ["TARGET-DOC"]

    retrieved_r1 = ["TARGET-DOC", "OTHER-1", "OTHER-2"]
    assert calculate_hit_at_k(retrieved_r1, expected, k=1) == 1.0
    assert calculate_hit_at_k(retrieved_r1, expected, k=3) == 1.0
    assert calculate_mrr(retrieved_r1, expected, k=3) == 1.0

    retrieved_r3 = ["OTHER-1", "OTHER-2", "TARGET-DOC", "OTHER-3"]
    assert calculate_hit_at_k(retrieved_r3, expected, k=1) == 0.0
    assert calculate_hit_at_k(retrieved_r3, expected, k=2) == 0.0
    assert calculate_hit_at_k(retrieved_r3, expected, k=3) == 1.0
    assert abs(calculate_mrr(retrieved_r3, expected, k=5) - (1.0 / 3.0)) < 1e-5

    retrieved_miss = ["OTHER-1", "OTHER-2", "OTHER-3"]
    assert calculate_hit_at_k(retrieved_miss, expected, k=3) == 0.0
    assert calculate_mrr(retrieved_miss, expected, k=3) == 0.0


def test_ood_three_way_decision_logic():
    """
    Menguji kerangka keputusan tiga arah:
    - REJECT: similarity < 0.840
    - ABSTAIN: 0.840 <= similarity <= 0.855 inclusive
    - ACCEPT: similarity > 0.855
    Memastikan bahwa kueri yang diabstensi TIDAK dihitung sebagai False Acceptance.
    """
    threshold = 0.840
    abstain_upper = 0.855

    def evaluate_decision(sim: float, is_ood: bool):
        if sim < threshold:
            sys_dec = "REJECT_AS_OOD"
        elif sim <= abstain_upper:
            sys_dec = "BORDERLINE_ABSTAIN"
        else:
            sys_dec = "ACCEPT_IN_DOMAIN"

        if is_ood:
            if sys_dec == "REJECT_AS_OOD":
                return "TRUE_REJECTION"
            elif sys_dec == "BORDERLINE_ABSTAIN":
                return "ABSTAIN"
            else:
                return "FALSE_ACCEPTANCE"
        else:
            if sys_dec == "ACCEPT_IN_DOMAIN":
                return "TRUE_ACCEPTANCE"
            elif sys_dec == "BORDERLINE_ABSTAIN":
                return "ABSTAIN"
            else:
                return "FALSE_REJECTION"

    # Kasus OOD di bawah threshold -> TRUE_REJECTION
    assert evaluate_decision(0.8162, is_ood=True) == "TRUE_REJECTION"

    # Kasus OOD dalam rentang abstensi (0.8447) -> ABSTAIN, BUKAN FALSE_ACCEPTANCE
    assert evaluate_decision(0.8447, is_ood=True) == "ABSTAIN"

    # Kasus OOD di atas rentang abstensi (0.8600) -> FALSE_ACCEPTANCE
    assert evaluate_decision(0.8600, is_ood=True) == "FALSE_ACCEPTANCE"

    # Kasus In-Domain di atas rentang abstensi -> TRUE_ACCEPTANCE
    assert evaluate_decision(0.8900, is_ood=False) == "TRUE_ACCEPTANCE"

    # Kasus In-Domain dalam rentang abstensi -> ABSTAIN
    assert evaluate_decision(0.8450, is_ood=False) == "ABSTAIN"

    # Kasus In-Domain di bawah threshold -> FALSE_REJECTION
    assert evaluate_decision(0.8100, is_ood=False) == "FALSE_REJECTION"


def test_benchmark_dataset_disjoint_splits_no_leakage():
    """
    Menguji bahwa 3 subset kueri benchmark (tuning, calibration, held-out evaluation)
    benar-benar terpisah (disjoint) tanpa kebocoran ID atau teks kueri.
    """
    benchmark_path = Path("data/evaluation/benchmark_queries.json")
    assert benchmark_path.exists(), "Berkas benchmark_queries.json tidak ditemukan!"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tuning = data.get("development_tuning_queries", [])
    calib = data.get("calibration_queries", [])
    eval_queries = data.get("held_out_evaluation_queries", [])

    assert len(tuning) >= 15, "Jumlah tuning queries minimal 15"
    assert len(calib) >= 10, "Jumlah calibration queries minimal 10"
    assert len(eval_queries) >= 30, "Jumlah held-out evaluation queries minimal 30"

    tuning_ids = {q["query_id"] for q in tuning}
    calib_ids = {q["query_id"] for q in calib}
    eval_ids = {q["query_id"] for q in eval_queries}

    assert len(tuning_ids & calib_ids) == 0, "Ditemukan kebocoran ID antara tuning dan calibration!"
    assert len(tuning_ids & eval_ids) == 0, "Ditemukan kebocoran ID antara tuning dan held-out evaluation!"
    assert len(calib_ids & eval_ids) == 0, "Ditemukan kebocoran ID antara calibration dan held-out evaluation!"

    tuning_texts = {q["query"].strip().lower() for q in tuning}
    calib_texts = {q["query"].strip().lower() for q in calib}
    eval_texts = {q["query"].strip().lower() for q in eval_queries}

    assert len(tuning_texts & calib_texts) == 0, "Ditemukan teks kueri identik antara tuning dan calibration!"
    assert len(tuning_texts & eval_texts) == 0, "Ditemukan teks kueri identik antara tuning dan held-out evaluation!"
    assert len(calib_texts & eval_texts) == 0, "Ditemukan teks kueri identik antara calibration dan held-out evaluation!"


def test_corpus_manifest_provenance_and_10_categories():
    """
    Menguji bahwa korpus memiliki tepat 10 kategori in-domain resmi,
    out_of_distribution tidak ada dalam taksonomi korpus,
    dan tidak ada klaim ko-publisher palsu.
    """
    manifest_path = Path("data/knowledge_base/manifest.json")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_10_categories = {
        "allergen_management",
        "catering_operations",
        "food_composition",
        "food_safety",
        "food_storage",
        "food_waste",
        "ingredient_substitution",
        "inventory_management",
        "menu_planning",
        "nutritional_standards",
    }

    manifest_categories = {doc["category"] for doc in manifest["documents"]}

    # Verifikasi tepat 10 kategori in-domain
    assert manifest_categories == expected_10_categories, (
        f"Kategori korpus tidak sesuai dengan 10 domain resmi: {manifest_categories}"
    )

    # Memastikan 'out_of_distribution' tidak ada dalam taksonomi kategori korpus
    assert "out_of_distribution" not in manifest_categories

    allowed_types = {"official_regulation", "national_guideline", "official_reference_data", "team_authored_sop"}
    for doc in manifest["documents"]:
        assert doc["document_type"] in allowed_types, f"Tipe dokumen tidak valid: {doc['document_type']}"

        if doc["document_type"] == "team_authored_sop":
            assert "simulation" in doc["issuer"].lower() or "nutrinexa" in doc["issuer"].lower()
            assert "persagi" not in doc["issuer"].lower()


def test_pgrs_single_source_integrity():
    """
    Memverifikasi bahwa Pedoman PGRS dan Permenkes 78/2013 mewakili satu sumber tunggal
    yang konsisten dan tidak dihitung ganda sebagai dua berkas berbeda.
    """
    manifest_path = Path("data/knowledge_base/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pgrs_docs = [d for d in manifest["documents"] if "pgrs" in d["source_id"].lower() or "pgrs" in d["source_path"].lower()]
    assert len(pgrs_docs) == 1, f"PGRS harus terdaftar tepat 1 kali di manifest, ditemukan {len(pgrs_docs)}"

    pgrs = pgrs_docs[0]
    assert pgrs["source_id"] == "PEDOMAN-KEMENKES-PGRS-2013"
    assert "78" in pgrs["source_title"]
    assert pgrs["category"] == "catering_operations"


def test_bpom_regulation_amendments_acknowledged():
    """
    Memverifikasi bahwa dokumen PerBPOM No. 31/2018 mengakui amandemen turunannya:
    PerBPOM No. 20/2021 dan PerBPOM No. 6/2024 beserta tautan sahnya.
    """
    bpom_path = Path("data/knowledge_base/bpom_manajemen_alergen_pangan.md")
    assert bpom_path.exists()
    content = bpom_path.read_text(encoding="utf-8")

    assert "PerBPOM No. 20/2021" in content or "20 Tahun 2021" in content
    assert "PerBPOM No. 6/2024" in content or "6 Tahun 2024" in content
    assert "jdih.pom.go.id" in content
