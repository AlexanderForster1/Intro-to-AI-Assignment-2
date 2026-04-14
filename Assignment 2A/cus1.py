from collections import deque

# Custom Search 1 - Bi-directional BFS algorithm (inspired by Bi-directional A*)

def cu1(node_list, origin, destination):

    OriginQueue = deque()
    GoalQueue = deque()
    visited1 = []
    visited2 = []

    OriginQueue.append([origin])
    visited1.append(origin)

    # To address pylance warnings for unbound sequences
    destination_node = None

    for destination_node in destination:
        GoalQueue.append([destination_node])
        visited2.append(destination_node)

    nodes_visited = 0

    while OriginQueue or GoalQueue:

        # To address pylance warnings for unbound sequences
        sequence1 = None 
        sequence2 = None
        
        ForwardPath = OriginQueue.popleft() if OriginQueue else None # To prevent possible crashing
        BackwardPath = GoalQueue.popleft() if GoalQueue else None
        
        if ForwardPath:
            sequence1 = ForwardPath[-1]
            nodes_visited += 1

            if sequence1 in destination:
                return sequence1, nodes_visited, ForwardPath
            
            for neighbour, cost in node_list[sequence1].edges:
                if neighbour not in visited1:
                    visited1.append(neighbour)
                    OriginQueue.append(ForwardPath + [neighbour])

        if BackwardPath:
            sequence2 = BackwardPath[-1]
            nodes_visited += 1
            
            if sequence2 == origin:
                return sequence2, nodes_visited, BackwardPath
            
            for neighbour, cost in node_list[sequence2].edges:
                if neighbour not in visited2:
                    visited2.append(neighbour)
                    GoalQueue.append(BackwardPath + [neighbour])     
        
        # Meeting Logic
        if ForwardPath and BackwardPath:
            if sequence1 in visited2 or sequence2 in visited1:
                meeting = sequence1 if sequence1 in visited2 else sequence2
                full_path = ForwardPath + BackwardPath[::-1]
        
                # Duplicate meeting node removal
                if full_path.count(meeting) > 1:
                    duplicate_node = full_path.index(meeting, 1)
                    full_path = full_path[:duplicate_node] + full_path[duplicate_node + 1:]

                return destination_node, nodes_visited, full_path

        if nodes_visited >= 78:
            return None, nodes_visited, []

    return None, nodes_visited, []

    