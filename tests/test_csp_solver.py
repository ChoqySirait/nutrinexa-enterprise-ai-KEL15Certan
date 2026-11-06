import pytest

from src.csp_solver import NutriNexaCSP, Recipe, backtracking_search


@pytest.fixture
def sample_recipes():
    return [
        Recipe("Sup Ayam Wortel", 400, {"Daging Ayam": 0.2, "Wortel": 0.1}, []),
        Recipe("Tumis Daging Sapi", 550, {"Daging Sapi": 0.2, "Wortel": 0.1}, []),
        Recipe("Omelet Tahu Bayam", 350, {"Tahu": 0.15, "Telur": 0.1}, ["Telur"]),
        Recipe("Ayam Bumbu Kecap", 500, {"Daging Ayam": 0.25}, []),
        Recipe("Salad Sayur Bening", 200, {"Wortel": 0.15}, []),
    ]


@pytest.fixture
def valid_csp_setup(sample_recipes):
    """Menyediakan instansiasi CSP yang valid dan memiliki solusi yang terbukti."""
    stok_dapur = {
        "Daging Ayam": 1.0,
        "Daging Sapi": 1.0,
        "Wortel": 0.5,
        "Tahu": 0.5,
        "Telur": 0.5,
    }
    variables = ['Makan_Pagi', 'Makan_Siang', 'Makan_Malam']
    domains = {v: list(sample_recipes) for v in variables}
    user_allergens = ["Telur"]
    expiring_items = ["Daging Ayam"]
    cal_min = 1000.0
    cal_max = 2000.0

    csp = NutriNexaCSP(
        variables=variables,
        domains=domains,
        inventory=stok_dapur,
        cal_min=cal_min,
        cal_max=cal_max,
        user_allergens=user_allergens,
        expiring_items=expiring_items,
    )
    solution = backtracking_search(csp)

    return {
        "csp": csp,
        "solution": solution,
        "variables": variables,
        "inventory": stok_dapur,
        "user_allergens": user_allergens,
        "expiring_items": expiring_items,
        "cal_min": cal_min,
        "cal_max": cal_max,
    }


def test_positive_valid_menu_returned(valid_csp_setup):
    """Skenario Positif 1: Kombinasi menu lengkap yang valid berhasil dikembalikan."""
    solution = valid_csp_setup["solution"]
    variables = valid_csp_setup["variables"]

    assert solution is not None
    assert len(solution) == len(variables)
    assert set(solution.keys()) == set(variables)
    assert all(isinstance(r, Recipe) for r in solution.values())


def test_positive_allergen_constraints_respected(valid_csp_setup):
    """Skenario Positif 2: Resep terpilih 100% bebas dari alergen pengguna."""
    solution = valid_csp_setup["solution"]
    user_allergens = set(valid_csp_setup["user_allergens"])

    assert solution is not None
    for slot, recipe in solution.items():
        assert not user_allergens.intersection(set(recipe.allergens)), (
            f"Resep '{recipe.name}' pada '{slot}' mengandung alergen terlarang!"
        )


def test_positive_recipes_not_duplicated(valid_csp_setup):
    """Skenario Positif 3: Anti-duplikasi (variasi menu terjaga)."""
    solution = valid_csp_setup["solution"]
    assert solution is not None

    recipe_names = [r.name for r in solution.values()]
    assert len(recipe_names) == len(set(recipe_names)), (
        f"Ditemukan resep duplikat: {recipe_names}"
    )


def test_positive_ingredient_consumption_respects_inventory(valid_csp_setup):
    """Skenario Positif 4: Total konsumsi bahan tidak melebihi stok fisik inventaris."""
    solution = valid_csp_setup["solution"]
    inventory = valid_csp_setup["inventory"]
    assert solution is not None

    total_used = {}
    for recipe in solution.values():
        for ing, qty in recipe.ingredients.items():
            total_used[ing] = total_used.get(ing, 0.0) + qty

    for ing, used_qty in total_used.items():
        assert used_qty <= inventory.get(ing, 0.0), (
            f"Penggunaan bahan '{ing}' ({used_qty} kg) melebihi stok inventaris ({inventory.get(ing, 0.0)} kg)"
        )


def test_positive_calories_respect_configured_range(valid_csp_setup):
    """Skenario Positif 5: Akumulasi kalori menu berada dalam batas minimum dan maksimum SOP."""
    solution = valid_csp_setup["solution"]
    cal_min = valid_csp_setup["cal_min"]
    cal_max = valid_csp_setup["cal_max"]
    assert solution is not None

    total_calories = sum(r.calories for r in solution.values())
    assert cal_min <= total_calories <= cal_max, (
        f"Total kalori ({total_calories} kcal) di luar rentang [{cal_min}, {cal_max}]"
    )


def test_positive_expiring_ingredients_used(valid_csp_setup):
    """Skenario Positif 6: Bahan pangan prioritas mendekati kadaluarsa berhasil dimanfaatkan."""
    solution = valid_csp_setup["solution"]
    inventory = valid_csp_setup["inventory"]
    expiring_items = valid_csp_setup["expiring_items"]
    assert solution is not None

    used_ingredients = set()
    for recipe in solution.values():
        used_ingredients.update(recipe.ingredients.keys())

    for item in expiring_items:
        if item in inventory and inventory[item] > 0:
            assert item in used_ingredients, (
                f"Bahan mendekati kadaluarsa '{item}' tidak digunakan dalam menu rekomendasi!"
            )


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


def test_order_domain_values_lcv_prioritizes_expiring_items(sample_recipes):
    """Menguji bahwa heuristik LCV menaruh resep yang memanfaatkan expiring_items di posisi pertama."""
    from src.csp_solver import order_domain_values_lcv

    inventory = {"Daging Ayam": 1.0, "Daging Sapi": 1.0, "Wortel": 1.0, "Tahu": 1.0, "Telur": 1.0}
    variables = ['Makan_Pagi', 'Makan_Siang']
    domains = {v: list(sample_recipes) for v in variables}

    # Set Daging Sapi sebagai expiring
    csp = NutriNexaCSP(variables, domains, inventory, 500, 2000, ["Telur"], ["Daging Sapi"])
    csp.enforce_unary_constraints()

    ordered = order_domain_values_lcv('Makan_Pagi', {}, csp)
    # Resep yang mengandung Daging Sapi (Tumis Daging Sapi) harus berada di urutan pertama
    assert "Daging Sapi" in ordered[0].ingredients
    assert ordered[0].name == "Tumis Daging Sapi"

