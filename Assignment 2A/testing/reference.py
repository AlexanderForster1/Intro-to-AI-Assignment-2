from utils import path_from_nx_edges
import networkx as nx

def dfs(G: nx.Graph, source: int, destinations: list[int]):
  destinations = set(destinations)
  # Run DFS
  nx_edges = nx.dfs_edges(G, source)
  return path_from_nx_edges(nx_edges, source, destinations)


def bfs(G: nx.Graph, source: int, destinations: list[int]):
  destinations = set(destinations)
  # Run BFS
  nx_edges = nx.bfs_edges(G, source)
  return path_from_nx_edges(nx_edges, source, destinations)

def shortest_path(G: nx.Graph, source: int, destinations: list[int]):
  destinations = set(destinations)

  # Compute all shortest simple paths from the given source in the graph
  # Returns a dictionary mapping each node to the shortest path to it from source
  lengths, paths = nx.single_source_dijkstra(G, source)

  print(paths)

  # Find the closest reachable destination node
  closest_node = None
  closest_path = []
  for d in destinations:
    if d in paths:
        if (not closest_path or lengths[d] < lengths[closest_node] or
            (lengths[d] == lengths[closest_node] and d < closest_node)):
          closest_node = d
          closest_path = paths[d]

  return closest_path

placeholder = lambda i, love, snoopy: []

reference_methods = {
  "DFS": dfs,
  "BFS": bfs,
  "GBFS": placeholder,
  "AS": shortest_path,
  "CUS1": placeholder,
  "CUS2": shortest_path,
}