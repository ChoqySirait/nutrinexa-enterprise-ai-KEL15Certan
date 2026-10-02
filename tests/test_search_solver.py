import pytest
from src.search_solver import a_star_search

def test_a_star_search_optimal_path():
    """Pengujian Unit untuk A* Search (Milestone 1)."""
    path, cost = a_star_search('Bahan_Awal', 'Target_Gizi')
    assert path == ['Bahan_Awal', 'Substitusi_A', 'Substitusi_C', 'Target_Gizi']
    assert cost == 7