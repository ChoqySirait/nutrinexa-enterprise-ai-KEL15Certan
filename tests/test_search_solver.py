import pytest

from src.search_solver import a_star_search


def test_a_star_search_optimal_path():
    """Pengujian Unit untuk A* Search jalur normal optimal (Milestone 1)."""
    path, cost = a_star_search('Bahan_Awal', 'Target_Gizi')
    assert path == ['Bahan_Awal', 'Substitusi_A', 'Substitusi_C', 'Target_Gizi']
    assert cost == 7


def test_a_star_search_start_equals_goal():
    """Kasus simpul awal sama dengan simpul tujuan."""
    path, cost = a_star_search('Bahan_Awal', 'Bahan_Awal')
    assert path == ['Bahan_Awal']
    assert cost == 0


def test_a_star_search_unreachable_goal(monkeypatch):
    """Kasus simpul tujuan tidak dapat dijangkau dari simpul awal."""
    custom_graph = {
        'Bahan_A': [('Bahan_B', 2)],
        'Bahan_B': [],
        'Target_Terisolasi': []
    }
    custom_heuristic = {
        'Bahan_A': 4,
        'Bahan_B': 2,
        'Target_Terisolasi': 0
    }
    monkeypatch.setattr('src.search_solver.graph', custom_graph)
    monkeypatch.setattr('src.search_solver.heuristic', custom_heuristic)

    path, cost = a_star_search('Bahan_A', 'Target_Terisolasi')
    assert path is None
    assert cost == float('inf')


def test_a_star_search_cyclic_graph_no_goal(monkeypatch):
    """Kasus graf memiliki siklus (loop) tanpa akses ke simpul tujuan."""
    cyclic_graph = {
        'Node_1': [('Node_2', 3)],
        'Node_2': [('Node_1', 3)],
        'Goal_Tak_Terjangkau': []
    }
    cyclic_heuristic = {
        'Node_1': 5,
        'Node_2': 5,
        'Goal_Tak_Terjangkau': 0
    }
    monkeypatch.setattr('src.search_solver.graph', cyclic_graph)
    monkeypatch.setattr('src.search_solver.heuristic', cyclic_heuristic)

    path, cost = a_star_search('Node_1', 'Goal_Tak_Terjangkau')
    assert path is None
    assert cost == float('inf')


def test_a_star_search_unknown_nodes_raise_value_error():
    """Kasus simpul awal atau tujuan tidak dikenal / tidak ada dalam ruang pencarian."""
    with pytest.raises(ValueError, match="tidak valid dalam ruang pencarian"):
        a_star_search('Simpul_Fiktif', 'Target_Gizi')

    with pytest.raises(ValueError, match="tidak valid dalam ruang pencarian"):
        a_star_search('Bahan_Awal', 'Simpul_Fiktif')


def test_build_food_substitution_graph_and_dynamic_search():
    """Menguji pembuatan graf pangan nyata dinamis dan pencarian rute substitusi dengan A*."""
    from src.search_solver import (
        DEFAULT_FOOD_CATALOG,
        build_food_substitution_graph,
        calculate_substitution_cost,
    )

    # 1. Uji katalog bahan pangan riil
    assert "Daging Sapi" in DEFAULT_FOOD_CATALOG
    assert "Tempe" in DEFAULT_FOOD_CATALOG

    # 2. Uji kalkulasi biaya substitusi
    cost = calculate_substitution_cost('Daging Sapi', 'Tempe')
    assert cost > 0

    with pytest.raises(ValueError, match="tidak terdaftar"):
        calculate_substitution_cost('Bahan_Aneh', 'Tempe')

    # 2. Bangun graf substitusi: Daging Sapi -> Daging Ayam -> Tempe -> Target Gizi Seimbang
    edges = [
        ('Daging Sapi', 'Daging Ayam'),
        ('Daging Sapi', 'Tempe'),
        ('Daging Ayam', 'Target Gizi Seimbang'),
        ('Tempe', 'Target Gizi Seimbang'),
    ]
    food_graph, food_heuristic = build_food_substitution_graph(
        edges=edges,
        target_node='Target Gizi Seimbang',
    )

    assert 'Daging Sapi' in food_graph
    assert food_heuristic['Target Gizi Seimbang'] == 0.0

    # 3. Jalankan pencarian A* pada graf pangan dinamis
    path, total_cost = a_star_search(
        start='Daging Sapi',
        goal='Target Gizi Seimbang',
        custom_graph=food_graph,
        custom_heuristic=food_heuristic,
    )

    assert path is not None
    assert path[0] == 'Daging Sapi'
    assert path[-1] == 'Target Gizi Seimbang'
    assert total_cost > 0
    assert total_cost < float('inf')

