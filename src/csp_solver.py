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

def select_unassigned_variable_mrv(assignment: Dict[str, Recipe], csp: NutriNexaCSP) -> str:
    """Heuristik MRV (Minimum Remaining Values): Pilih variabel dengan sisa domain terkecil."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda var: len(csp.domains[var]))


def is_consistent(var: str, recipe: Recipe, assignment: Dict[str, Recipe], csp: NutriNexaCSP) -> bool:
    """Memeriksa konsistensi penugasan 'recipe' ke 'var' terhadap seluruh batasan bisnis."""
    temp_assignment = deepcopy(assignment)
    temp_assignment[var] = recipe

    # 1. Cek C4: Variasi Menu (Anti-Duplikasi)
    assigned_names = [r.name for r in temp_assignment.values()]
    if len(assigned_names) != len(set(assigned_names)):
        return False

    # 2. Cek C2: Akumulasi Stok Inventaris
    total_used: Dict[str, float] = {}
    for r in temp_assignment.values():
        for ing, qty in r.ingredients.items():
            total_used[ing] = total_used.get(ing, 0.0) + qty
            if total_used[ing] > csp.inventory.get(ing, 0.0):
                return False

    # 3. Cek C3: Batas Kalori Maksimum
    current_cal = sum(r.calories for r in temp_assignment.values())
    if current_cal > csp.cal_max:
        return False

    # 4. Jika Seluruh Variabel Terisi, Cek Kalori Minimum & Bahan Kadaluarsa (C5)
    if len(temp_assignment) == len(csp.variables):
        if current_cal < csp.cal_min:
            return False

        used_ingredients = set(total_used.keys())
        for exp_item in csp.expiring_items:
            if exp_item in csp.inventory and csp.inventory[exp_item] > 0:
                if exp_item not in used_ingredients or total_used[exp_item] <= 0:
                    return False

    return True


def backtracking_search(csp: NutriNexaCSP) -> Optional[Dict[str, Recipe]]:
    """Eksekusi utama pencarian Backtracking dipadu dengan AC-3 dan MRV."""
    csp.enforce_unary_constraints()
    if not ac3(csp):
        return None  # Pra-pemrosesan AC-3 mendeteksi pertentangan batasan
    return backtrack({}, csp)


def backtrack(assignment: Dict[str, Recipe], csp: NutriNexaCSP) -> Optional[Dict[str, Recipe]]:
    """Fungsi rekursif Backtracking."""
    if len(assignment) == len(csp.variables):
        return assignment

    csp.backtrack_count += 1
    var = select_unassigned_variable_mrv(assignment, csp)

    for recipe in csp.domains[var]:
        if is_consistent(var, recipe, assignment, csp):
            assignment[var] = recipe
            result = backtrack(assignment, csp)
            if result is not None:
                return result
            del assignment[var]  # Backtracking

    return None
