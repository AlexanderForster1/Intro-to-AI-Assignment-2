from collections import deque

def bfs(node_list, origin, destination):

    # Defensive coding to prevent crash from missing origin
    if origin not in node_list:
        return None, 0, []

    queue = deque()
    visited = set()

    queue.append([origin])
    visited.add(origin)
    nodes_visited = 0

    while queue:
        path = queue.popleft()
        sequence = path[-1]
        #print(sequence, end = " ")
        nodes_visited += 1

        if sequence in destination:
            #print(f"\nDestination found! Path length: {nodes_visited}")
            return sequence, nodes_visited, path

        for neighbour, cost in node_list[sequence].edges:
            if neighbour not in visited:
                visited.add(neighbour)
                #queue.append(neighbour)
                queue.append(path + [neighbour])

    return None, nodes_visited, []

    
    