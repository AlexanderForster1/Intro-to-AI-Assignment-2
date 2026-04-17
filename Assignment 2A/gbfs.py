import heuristics
import heapq

def gbfs(node_list, origin, destination):
    Node_heuristic = []
    heapq.heappush(Node_heuristic, (0, origin))
    Node_count = 0
    visited = set()
    parent = {origin: None}

    while Node_heuristic:
        current_node = heapq.heappop(Node_heuristic)[1]
        if current_node in visited:
            continue
        visited.add(current_node)
        Node_count += 1

        if current_node in destination:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()
            return path[-1], Node_count, path
        for neighbor, cost in node_list[current_node].edges:
            if neighbor not in visited and neighbor not in parent:
                heuristic_costs = [heuristics.manhattan_distance(node_list[neighbor], node_list[d]) for d in destination]
                heuristic_cost  = min(heuristic_costs) 
                heapq.heappush(Node_heuristic, (heuristic_cost, neighbor))
                parent[neighbor] = current_node
    return  None, Node_count, []