import sys
import os
import json
import heapq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Assignment 2A'))

from graph_builder import build_graph, Node
from travel_time import travel_time_seconds

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def build_travel_time_graph(
    flow_dict: dict[int, float] | None = None,
) -> tuple[dict[int, Node], dict[int, dict]]:
    """Build the Boroondara graph with edge costs set to travel time in seconds."""
    node_list, sites = build_graph()

    if flow_dict is None:
        flow_dict = {}

    for scats, node in node_list.items():
        flow = flow_dict.get(scats, 0.0)
        node.edges = [
            (nb, travel_time_seconds(dist_km, flow))
            for nb, dist_km in node.edges
        ]

    return node_list, sites


def _astar(
    node_list: dict[int, Node],
    origin: int,
    destinations: list[int],
) -> tuple[int | None, list[int]]:
    """A* on the travel-time graph. Returns (goal, path)."""
    dest_set = set(destinations)
    valid_dest = [d for d in destinations if d in node_list]

    def h(node_id: int) -> float:
        ox, oy = node_list[node_id].coordinates
        return min(
            ((ox - node_list[d].coordinates[0]) ** 2
             + (oy - node_list[d].coordinates[1]) ** 2) ** 0.5
            for d in valid_dest
        )

    g_cost = {origin: 0.0}
    predecessor = {origin: None}
    frontier = [(h(origin), origin)]

    while frontier:
        f, current = heapq.heappop(frontier)

        if current in dest_set:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return current, list(reversed(path))

        if f > g_cost.get(current, float('inf')) + h(current) + 1e-9:
            continue

        for neighbour, cost in node_list[current].edges:
            new_g = g_cost[current] + cost
            if new_g < g_cost.get(neighbour, float('inf')):
                g_cost[neighbour] = new_g
                predecessor[neighbour] = current
                heapq.heappush(frontier, (new_g + h(neighbour), neighbour))

    return None, []


def _path_cost(node_list: dict[int, Node], path: list[int]) -> float:
    """Sum edge costs along a path."""
    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for nb, cost in node_list[a].edges:
            if nb == b:
                total += cost
                break
    return total


def yen_k_shortest(
    node_list: dict[int, Node],
    origin: int,
    destination: int,
    k: int = 5,
) -> list[list[int]]:
    """
    Yen's k-shortest loopless paths algorithm layered on top of A*.
    Returns a list of up to k paths (each a list of SCATS numbers).
    """
    goal, first_path = _astar(node_list, origin, [destination])
    if not first_path:
        return []

    found = [first_path]
    candidates: list[tuple[float, list[int]]] = []

    for _ in range(k - 1):
        prev_path = found[-1]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            removed_edges: list[tuple[int, int, float]] = []

            for path in found:
                if path[:i + 1] == root_path and i + 1 < len(path):
                    a, b = path[i], path[i + 1]
                    for idx, (nb, cost) in enumerate(node_list[a].edges):
                        if nb == b:
                            node_list[a].edges.pop(idx)
                            removed_edges.append((a, b, cost))
                            break

            removed_node_edges: dict[int, list] = {}
            for root_node in root_path[:-1]:
                removed_node_edges[root_node] = node_list[root_node].edges[:]
                node_list[root_node].edges = []

            _, spur_path = _astar(node_list, spur_node, [destination])

            if spur_path:
                total_path = root_path[:-1] + spur_path
                cost = _path_cost(node_list, total_path)
                if total_path not in [p for _, p in candidates] and total_path not in found:
                    heapq.heappush(candidates, (cost, total_path))

            for a, b, cost in removed_edges:
                node_list[a].edges.append((b, cost))
                node_list[a].edges.sort(key=lambda e: e[0])

            for root_node, edges in removed_node_edges.items():
                node_list[root_node].edges = edges

        if not candidates:
            break

        _, best = heapq.heappop(candidates)
        found.append(best)

    return found


def _run_algorithm(
    algorithm: str,
    node_list: dict[int, Node],
    origin: int,
    destination: int,
) -> list[int]:
    """
    Run one of the six 2A algorithms and return the path.
    Imports are deferred so the sys.path insert above takes effect first.
    """
    dest = [destination]

    if algorithm == 'AS':
        _, _, path = __import__('astar').astar(node_list, origin, dest)

    elif algorithm == 'BFS':
        _, _, path = __import__('bfs').bfs(node_list, origin, dest)

    elif algorithm == 'DFS':
        import input as _input_mod
        sys.modules.setdefault('input', _input_mod)
        _, _, path = __import__('dfs').dfs(node_list, origin, dest)

    elif algorithm == 'GBFS':
        _, _, path = __import__('gbfs').gbfs(node_list, origin, dest)

    elif algorithm == 'CUS1':
        _, _, path = __import__('cus1').cus1(node_list, origin, dest)

    elif algorithm == 'CUS2':
        import input as _input_mod
        sys.modules.setdefault('input', _input_mod)
        _, _, path = __import__('cus2').cus2(node_list, origin, dest)

    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    return path


def find_routes(
    origin: int,
    destination: int,
    flow_dict: dict[int, float] | None = None,
    algorithm: str = 'AS',
    max_routes: int | None = None,
) -> list[dict]:
    """
    Find up to max_routes paths from origin to destination.

    For AS (A*), uses Yen's k-shortest paths to return multiple routes.
    For all other algorithms, returns a single route.
    """
    config = load_config()
    if max_routes is None:
        max_routes = config.get('max_routes', 5)

    node_list, sites = build_travel_time_graph(flow_dict)

    if origin not in node_list or destination not in node_list:
        return []

    if algorithm == 'AS':
        paths = yen_k_shortest(node_list, origin, destination, k=max_routes)
    else:
        path = _run_algorithm(algorithm, node_list, origin, destination)
        paths = [path] if path else []

    results = []
    for path in paths:
        cost_s = _path_cost(node_list, path)
        results.append({
            'path'         : path,
            'cost_seconds' : cost_s,
            'cost_minutes' : cost_s / 60.0,
            'sites'        : sites,
        })

    return results