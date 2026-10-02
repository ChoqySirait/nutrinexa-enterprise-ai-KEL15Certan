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