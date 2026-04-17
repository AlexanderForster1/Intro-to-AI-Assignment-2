import heuristics
def gbfs(node_list, origin, destination):
    Node_heuristic = []
    Node_heuristic.append((origin, 0))
    Node_count = 0
    visited = set()
    parent = {origin: None}
    goal = destination[0] if isinstance(destination, list) else destination

    while Node_heuristic:
        Node_heuristic.sort(key=lambda x: x[1])
        current_node = Node_heuristic.pop(0)[0]
        if current_node in visited:
            continue
        visited.add(current_node)
        Node_count += 1


        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
                path.reverse()
            return goal,  Node_count, path
        for neighbor, cost in node_list[current_node].edges:
            if neighbor not in visited and neighbor not in parent:
                heuristic_cost = heuristics.manhattan_distance(node_list[neighbor], node_list[goal])
                Node_heuristic.append((neighbor, heuristic_cost))
                parent[neighbor] = current_node
    return  None, Node_count, []