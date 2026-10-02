# NutriNexa (CERTAN-KEL-15)

NutriNexa (*Enterprise Food Waste & Recipe Optimizer*) adalah Sistem Informasi Cerdas Enterprise berbasis *Agentic RAG* yang dirancang untuk membantu pengelolaan bahan pangan, pencarian informasi SOP gizi, serta optimasi rekomendasi resep *zero-waste*.

NutriNexa mengintegrasikan beberapa komponen utama:
1. **Agentic RAG Orchestrator & Knowledge Base (Vector DB):** Untuk pencarian kontekstual dokumen SOP gizi internal.
2. **Modul Input Data Stok & Parameter Gizi:** Untuk pemrosesan data inventaris bahan pangan terstruktur (JSON/CSV) dan teks kueri interaktif.
3. **Baseline A* Search Engine (`src/search_solver.py`):** Algoritma pencarian berbasis *cost & heuristic* ($f(n) = g(n) + h(n)$) untuk menentukan rute substitusi bahan terhemat (Milestone 1).
4. **CSP Optimization Engine (`src/csp_solver.py`):** Mesin pemecah batasan berbasis *Constraint Satisfaction Problem* (AC-3 Propagation & Backtracking Search dengan MRV Heuristic) untuk menyusun alokasi menu harian yang patuh 100% pada batasan stok, alergi, dan batas kalori SOP (Milestone 2).

---

## 👥 Tim Pengembang (Kelompok 15)
- **AI Architect & Model Lead:** Choqy Pananda Sirait (12S24012)
- **Data & Knowledge Engineer:** Yesika Nadia Saragih (12S24024)
- **Integration & Interface Engineer:** Josua Sianturi (12S24035)
- **QA, Evaluation & Ethics Lead:** Jaya Bestina Simbolon (12S24023)

---

## 📐 Arsitektur Sistem (System Architecture)

Diagram arsitektur sistem NutriNexa menunjukkan integrasi antara antarmuka pengguna, *Agentic RAG*, *Modul Input*, serta mesin optimasi resep berbasis *A* Search* (`src/search_solver.py`) dan *CSP Solver* (`src/csp_solver.py`):

```mermaid
graph TD
    User([Pengguna / Enterprise Client]) -->|Request / Konsultasi| UI[X-Platform Interface / API Gateway]
    
    subgraph NutriNexa Core Enterprise AI Engine
        UI --> Agent[Agentic RAG Orchestrator]
        Agent --> Knowledge[Knowledge Base / Vector DB]
        Agent --> InputMod[Modul Input]
        Agent --> Optimizer[Optimasi Resep Zero-Waste & A* Search]
    end

    Optimizer -->|"Algoritma A* (Cost & Heuristic)"| Solver["src/search_solver.py"]
    Optimizer -->|"CSP Engine (AC-3 & MRV)"| CSPSolver["src/csp_solver.py"]
    
    Solver --> Output[Rute Substitusi Pangan & Rekomendasi Gizi]
    CSPSolver --> Output
    
    Output -.->|Response| UI