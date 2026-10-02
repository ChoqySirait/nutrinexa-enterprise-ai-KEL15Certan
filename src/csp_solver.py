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

<<<<<<< HEAD
            
=======
    def ac3(csp: NutriNexaCSP) -> bool:
    """
    Algoritma Arc Consistency (AC-3) untuk memangkas domain yang tidak konsisten.
    """
    queue: List[Tuple[str, str]] = [(xi, xj) for xi in csp.variables for xj in csp.variables if xi != xj]

    while queue:
        xi, xj = queue.pop(0)
        if revise(csp, xi, xj):
            if len(csp.domains[xi]) == 0:
                return False  # Domain kosong: Tidak ada solusi yang memenuhi batasan
            for xk in csp.variables:
                if xk != xi and xk != xj:
                    queue.append((xk, xi))
    return True


def revise(csp: NutriNexaCSP, xi: str, xj: str) -> bool:
    """Revisi domain Xi jika tidak ada nilai di Xj yang mendukung batasan C2 dan C4."""
    revised = False
    to_remove = []

    for r_i in csp.domains[xi]:
        has_support = False
        for r_j in csp.domains[xj]:
            if r_i.name != r_j.name:  # C4: Variasi Menu
                valid_stock = True
                for ing, qty in r_i.ingredients.items():
                    total_req = qty + r_j.ingredients.get(ing, 0.0)
                    if total_req > csp.inventory.get(ing, 0.0):  # C2: Batas Stok
                        valid_stock = False
                        break
                if valid_stock:
                    has_support = True
                    break

        if not has_support:
            to_remove.append(r_i)
            revised = True

    for r in to_remove:
        csp.domains[xi].remove(r)
        csp.pruned_nodes_ac3 += 1

    return revised
>>>>>>> 13b319623919a75d1989f70de2f6315fa1ed9c53
