import heapq

graph = {
    'Bahan_Awal': [('Substitusi_A', 2), ('Substitusi_B', 5)],
    'Substitusi_A': [('Substitusi_C', 4), ('Target_Gizi', 7)],
    'Substitusi_B': [('Target_Gizi', 2)],
    'Substitusi_C': [('Target_Gizi', 1)],
    'Target_Gizi': []
}

heuristic = {
    'Bahan_Awal': 5,
    'Substitusi_A': 3,
    'Substitusi_B': 2,
    'Substitusi_C': 1,
    'Target_Gizi': 0
}

def a_star_search(start, goal):
    """
    Algoritma A* Search untuk Baseline Search NutriNexa (Milestone 1)
    """
    pq = []
    # (f_score, g_score, current_node, path)
    heapq.heappush(pq, (heuristic[start], 0, start, [start]))
    visited = {}

    while pq:
        f_score, g_score, current, path = heapq.heappop(pq)

        if current == goal:
            return path, g_score

        if current in visited and visited[current] <= g_score:
            continue
        visited[current] = g_score

        for neighbor, cost in graph.get(current, []):
            tentative_g = g_score + cost
            tentative_f = tentative_g + heuristic[neighbor]
            heapq.heappush(pq, (tentative_f, tentative_g, neighbor, path + [neighbor]))

    return None, float('inf')

if __name__ == "__main__":
    start_node = 'Bahan_Awal'
    target_node = 'Target_Gizi'
    path, cost = a_star_search(start_node, target_node)
    print(f"Rute Optimasi Substitusi Pangan Terbaik: {' -> '.join(path)}")
    print(f"Total Biaya/Cost Nutrisi Terendah: {cost}") 

import pytest
from src.csp_solver import Recipe, NutriNexaCSP, backtracking_search

@pytest.fixture
def sample_recipes():
    return [
        Recipe("Sup Ayam Wortel", 400, {"Daging Ayam": 0.2, "Wortel": 0.1}, []),
        Recipe("Tumis Daging Sapi", 550, {"Daging Sapi": 0.2, "Wortel": 0.1}, []),
        Recipe("Omelet Tahu Bayam", 350, {"Tahu": 0.15, "Telur": 0.1}, ["Telur"]),
        Recipe("Ayam Bumbu Kecap", 500, {"Daging Ayam": 0.25}, []),
        Recipe("Salad Sayur Bening", 200, {"Wortel": 0.15}, [])
    ]

def test_extreme_case_overconstrained_inventory(sample_recipes):
    """Kasus Ekstrem 1: Stok bahan baku 0 kg (Over-constrained). Must Return None Instantly."""
    inventory = {"Daging Ayam": 0.0, "Daging Sapi": 0.0, "Tahu": 0.0, "Wortel": 0.0}
    variables = ['Makan_Pagi', 'Makan_Siang', 'Makan_Malam']
    domains = {v: list(sample_recipes) for v in variables}
    
    csp = NutriNexaCSP(variables, domains, inventory, 1000, 2000, [], [])
    solution = backtracking_search(csp)
    
    assert solution is None  # Terbukti terkonvergensi gagal secara aman tanpa error

def test_extreme_case_strict_allergen(sample_recipes):
    """Kasus Ekstrem 2: Alergi memangkas seluruh isi domain variabel."""
    inventory = {"Daging Ayam": 1.0, "Wortel": 1.0, "Tahu": 1.0, "Telur": 1.0}
    variables = ['Makan_Pagi']
    domains = {'Makan_Pagi': [sample_recipes[2]]}  # Hanya Omelet (Telur)
    
    csp = NutriNexaCSP(variables, domains, inventory, 200, 800, ["Telur"], [])
    solution = backtracking_search(csp)
    
    assert solution is None