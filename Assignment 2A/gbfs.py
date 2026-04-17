import heuristics
def gbfs(node_list, destination, origin):
    Node_heuristic = []
    Node_heuristic.append((origin, 0))
    Node_count = 0
    visited = set()
    while Node_heuristic:
        Node_heuristic.sort(key=lambda x: x[1])
        current_node = Node_heuristic.pop(0)[0]
        Node_count += 1
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node in destination:
            return current_node,  Node_count, visited
        for neighbor, cost in node_list[current_node].edges:
            if neighbor not in visited:
                heuristic_cost = heuristics.euclidean_distance(node_list[neighbor], node_list[destination[0]])
                Node_heuristic.append((neighbor, heuristic_cost))
    return  None, Node_count, visited