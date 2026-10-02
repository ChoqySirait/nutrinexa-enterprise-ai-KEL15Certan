"""
NutriNexa CSP Solver Module 
Mengimplementasikan Constraint Satisfaction Problem (CSP) dengan algoritma AC-3
(Arc Consistency) dan Backtracking Search menggunakan heuristik MRV (Minimum Remaining Values).
"""

from typing import Dict, List, Set, Optional, Tuple
from copy import deepcopy


class Recipe:
    """Kelas untuk merepresentasikan entitas resep dan kandungan gizinya."""
    def __init__(self, name: str, calories: float, ingredients: Dict[str, float], allergens: List[str]):
        self.name = name
        self.calories = calories
        self.ingredients = ingredients  # Contoh: {"Daging Ayam": 0.2, "Wortel": 0.1} (dalam kg)
        self.allergens = allergens

    def __repr__(self):
        return f"Recipe('{self.name}', {self.calories} kcal)"


class NutriNexaCSP:
    """Mesin inferensi CSP NutriNexa untuk alokasi menu terikat batasan bisnis."""
    def __init__(self, variables: List[str], domains: Dict[str, List[Recipe]], inventory: Dict[str, float],
                 cal_min: float, cal_max: float, user_allergens: List[str], expiring_items: List[str]):
        self.variables = variables  # Contoh: ['Makan_Pagi', 'Makan_Siang', 'Makan_Malam']
        self.domains = domains      # Dict[var, List[Recipe]]
        self.inventory = inventory  # Dict[nama_bahan, stok_kg]
        self.cal_min = cal_min
        self.cal_max = cal_max
        self.user_allergens = set(user_allergens)
        self.expiring_items = set(expiring_items)
        
        # Metrik performa untuk analisis konvergensi
        self.backtrack_count = 0
        self.pruned_nodes_ac3 = 0

    def enforce_unary_constraints(self):
        """Memangkas domain awal berdasarkan batasan alergi (C1: Unary Constraint)."""
        for var in self.variables:
            valid_recipes = []
            for recipe in self.domains[var]:
                if not self.user_allergens.intersection(set(recipe.allergens)):
                    valid_recipes.append(recipe)
            self.domains[var] = valid_recipes