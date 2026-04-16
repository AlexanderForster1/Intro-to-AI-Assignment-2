import heapq
from input import Node
from heuristics import euclidean_distance

def astar(node_list: dict[int, Node], src: int, dest: list[int]):
    '''
    A* search algorithm using euclidean distance as the heuristic h(n).
    Expands nodes by lowest f(n) = g(n) + h(n), where g(n) is the cost from
    the source to the current node and h(n) is the minimum euclidean distance
    to any destination. Ties are broken by ascending node number. Returns a
    tuple of (goal, number_of_nodes_created, path).
    '''
    if node_list.get(src) is None or not (set(dest) & set(node_list)):
        raise ValueError("Source or destination node not in graph")

    dest_set = set(dest)
    valid_dest = [d for d in dest if d in node_list]

    def h(node_number):
        return min(euclidean_distance(node_list[node_number], node_list[d]) for d in valid_dest)

    g_cost = {src: 0}
    predecessor = {src: None}
    nodes_created = set()
    nodes_created.add(src)

    frontier = [(h(src), src)]

    while frontier:
        f, current = heapq.heappop(frontier)

        if current in dest_set:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return (current, len(nodes_created), list(reversed(path)))

        if f > g_cost.get(current, float('inf')) + h(current) + 1e-9:
            continue

        for neighbour, cost in node_list[current].edges:
            new_g = g_cost[current] + cost
            if new_g < g_cost.get(neighbour, float('inf')):
                g_cost[neighbour] = new_g
                predecessor[neighbour] = current
                heapq.heappush(frontier, (new_g + h(neighbour), neighbour))
                nodes_created.add(neighbour)

    return (None, len(nodes_created), [])