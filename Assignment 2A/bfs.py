from collections import deque

def bfs(node_list, origin, destination):

    queue = deque()
    visited = []

    queue.append([origin])
    visited.append(origin)
    nodes_visited = 0

    while queue:
        path = queue.popleft()
        sequence = path[-1]
        #print(sequence, end = " ")

        if sequence in destination:
            #print(f"\nDestination found! Path length: {nodes_visited}")
            return sequence, nodes_visited, path

        if nodes_visited >= 78:
            #print(f"\nBFS has not found the node, it has visisted: {visited}")
            return None, nodes_visited, []

        for neighbour, cost in node_list[sequence].edges:
            if neighbour not in visited:
                visited.append(neighbour)
                #queue.append(neighbour)
                queue.append(path + [neighbour])
        
        nodes_visited += 1

    return None, nodes_visited, []

    
    