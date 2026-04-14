def manhattan_distance(node1, node2):
    x1, y1 = node1.coordinates
    x2, y2 = node2.coordinates
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(node1, node2):
    x1, y1 = node1.coordinates
    x2, y2 = node2.coordinates
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def chebyshev_distance(node1, node2):
    x1, y1 = node1.coordinates
    x2, y2 = node2.coordinates
    return max(abs(x1 - x2), abs(y1 - y2))
