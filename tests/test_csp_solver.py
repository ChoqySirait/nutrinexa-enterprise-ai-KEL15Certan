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
    """Kasus Ekstrem 1: Stok bahan baku 0 kg (Must Return None Instantly)."""
    inventory = {"Daging Ayam": 0.0, "Daging Sapi": 0.0, "Tahu": 0.0, "Wortel": 0.0}
    variables = ['Makan_Pagi', 'Makan_Siang', 'Makan_Malam']
    domains = {v: list(sample_recipes) for v in variables}
    
    csp = NutriNexaCSP(variables, domains, inventory, 1000, 2000, [], [])
    solution = backtracking_search(csp)
    assert solution is None

def test_extreme_case_strict_allergen(sample_recipes):
    """Kasus Ekstrem 2: Alergi memangkas seluruh isi domain."""
    inventory = {"Daging Ayam": 1.0, "Daging Sapi": 1.0, "Tahu": 1.0, "Telur": 1.0}
    variables = ['Makan_Pagi']
    domains = {'Makan_Pagi': [sample_recipes[2]]}
    
    csp = NutriNexaCSP(variables, domains, inventory, 200, 800, ["Telur"], [])
    solution = backtracking_search(csp)
    assert solution is None