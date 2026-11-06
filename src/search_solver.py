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

# Katalog Bahan Pangan Riil Default (Nutrisi per 100g dan estimasi biaya per porsi)
DEFAULT_FOOD_CATALOG = {
    'Daging Sapi': {'kalori': 250.0, 'protein': 26.0, 'biaya': 25.0},
    'Daging Ayam': {'kalori': 165.0, 'protein': 31.0, 'biaya': 12.0},
    'Ikan Kembung': {'kalori': 125.0, 'protein': 21.0, 'biaya': 10.0},
    'Tempe': {'kalori': 190.0, 'protein': 19.0, 'biaya': 4.0},
    'Tahu': {'kalori': 80.0, 'protein': 8.0, 'biaya': 3.0},
    'Jamur Tiram': {'kalori': 35.0, 'protein': 3.0, 'biaya': 5.0},
    'Target Gizi Seimbang': {'kalori': 150.0, 'protein': 20.0, 'biaya': 0.0},
}


def calculate_substitution_cost(
    item_a: str,
    item_b: str,
    catalog: dict[str, dict[str, float]] | None = None,
    weight_cost: float = 0.5,
    weight_nutrition: float = 0.5,
) -> float:
    """
    Menghitung biaya transisi substitusi bahan pangan berdasarkan selisih harga dan profil nutrisi.
    """
    active_catalog = catalog if catalog is not None else DEFAULT_FOOD_CATALOG
    if item_a not in active_catalog or item_b not in active_catalog:
        raise ValueError(f"Bahan '{item_a}' atau '{item_b}' tidak terdaftar dalam katalog pangan.")

    data_a = active_catalog[item_a]
    data_b = active_catalog[item_b]

    cost_diff = abs(data_a['biaya'] - data_b['biaya'])
    nutrition_diff = abs(data_a['protein'] - data_b['protein']) * 0.5 + abs(data_a['kalori'] - data_b['kalori']) * 0.05
    return round(weight_cost * cost_diff + weight_nutrition * nutrition_diff, 2)


def build_food_substitution_graph(
    edges: list[tuple[str, str]],
    target_node: str,
    catalog: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    """
    Membangun graf substitusi bahan pangan nyata dinamis beserta estimasi biaya dan nilai heuristik.
    """
    active_catalog = catalog if catalog is not None else DEFAULT_FOOD_CATALOG
    if target_node not in active_catalog:
        raise ValueError(f"Target node '{target_node}' tidak ditemukan dalam katalog bahan pangan.")

    dynamic_graph: dict[str, list[tuple[str, float]]] = {}
    nodes: set[str] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    nodes.add(target_node)

    for node in nodes:
        dynamic_graph[node] = []

    for u, v in edges:
        cost = calculate_substitution_cost(u, v, active_catalog)
        dynamic_graph[u].append((v, cost))

    dynamic_heuristic: dict[str, float] = {}
    target_data = active_catalog[target_node]
    for node in nodes:
        if node == target_node:
            dynamic_heuristic[node] = 0.0
        else:
            n_data = active_catalog[node]
            h_val = abs(n_data['protein'] - target_data['protein']) * 0.2 + abs(n_data['kalori'] - target_data['kalori']) * 0.02
            dynamic_heuristic[node] = round(h_val, 2)

    return dynamic_graph, dynamic_heuristic


def a_star_search(start, goal, custom_graph=None, custom_heuristic=None):
    """
    Algoritma A* Search untuk Baseline Search NutriNexa (Milestone 1).
    Mendukung graf kustom dinamis untuk perutean substitusi pangan nyata.
    """
    active_graph = custom_graph if custom_graph is not None else graph
    active_heuristic = custom_heuristic if custom_heuristic is not None else heuristic

    if start not in active_heuristic or goal not in active_heuristic:
        raise ValueError(
            f"Simpul awal '{start}' atau simpul tujuan '{goal}' tidak valid dalam ruang pencarian."
        )

    pq = []
    # (f_score, g_score, current_node, path)
    heapq.heappush(pq, (active_heuristic[start], 0, start, [start]))
    visited = {}

    while pq:
        _f_score, g_score, current, path = heapq.heappop(pq)

        if current == goal:
            return path, g_score

        if current in visited and visited[current] <= g_score:
            continue
        visited[current] = g_score

        for neighbor, cost in active_graph.get(current, []):
            if neighbor not in active_heuristic:
                raise ValueError(
                    f"Simpul tetangga '{neighbor}' tidak valid dalam ruang pencarian."
                )
            tentative_g = g_score + cost
            tentative_f = tentative_g + active_heuristic[neighbor]
            heapq.heappush(pq, (tentative_f, tentative_g, neighbor, path + [neighbor]))

    return None, float('inf')

if __name__ == "__main__":
    start_node = 'Bahan_Awal'
    target_node = 'Target_Gizi'
    path, cost = a_star_search(start_node, target_node)
    print(f"Rute Optimasi Substitusi Pangan Terbaik: {' -> '.join(path)}")
    print(f"Total Biaya/Cost Nutrisi Terendah: {cost}")