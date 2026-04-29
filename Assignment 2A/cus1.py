from collections import deque

# Custom Search 1 - Unsupervised Bi-directional BFS algorithm (inspired by Bi-directional A*)

def cus1(node_list, origin, destination):

    # Defensive coding to prevent crashes from missing origin or destination
    if origin not in node_list:
        return None, 0, []
    valid_destinations = [d for d in destination if d in node_list]
    if not valid_destinations:
        return None, 0, []
    
    # Reversal for bi-direction search
    reverse_map = {node: [] for node in node_list}
    for node_id in node_list:
        for neighbour, cost in node_list[node_id].edges:
            reverse_map[neighbour].append(node_id)

    # Ditctionary for path storage
    visited_from_origin = {origin: [origin]}
    visited_from_goal = {d: [d] for d in destination}

    origin_queue = deque([origin])
    goal_queue = deque(destination)

    nodes_visited = 0

    while origin_queue and goal_queue:
        # Forward Search
        forwards_search = origin_queue.popleft()
        nodes_visited += 1

        # Check if the destination is reached directly (Standard BFS check)
        if forwards_search in destination:
            return forwards_search, nodes_visited, visited_from_origin[forwards_search]

        for neighbour, _ in node_list[forwards_search].edges:
            if neighbour not in visited_from_origin:
                # Store the path to this neighbor
                visited_from_origin[neighbour] = visited_from_origin[forwards_search] + [neighbour]
                
                # Meetign logic: If the backward search already found this node
                if neighbour in visited_from_goal:
                    full_path = visited_from_origin[neighbour] + visited_from_goal[neighbour][::-1][1:]
                    return full_path[-1], nodes_visited, full_path
                
                origin_queue.append(neighbour)

        # Backwards Search
        backwards_search = goal_queue.popleft()
        nodes_visited += 1

        # Reverse Search
        for parent in reverse_map[backwards_search]:
            if parent not in visited_from_goal:
                # Destination searching for Origin direction
                visited_from_goal[parent] = visited_from_goal[backwards_search] + [parent]

                # Meeting Logic: If the forward search already found this node
                if parent in visited_from_origin:
                    full_path = visited_from_origin[parent] + visited_from_goal[parent][::-1][1:]
                    return full_path[-1], nodes_visited, full_path
                
                goal_queue.append(parent)

    return None, nodes_visited, []