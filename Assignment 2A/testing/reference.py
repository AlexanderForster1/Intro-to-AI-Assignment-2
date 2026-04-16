import math
import networkx as nx
from utils import path_from_nx_edges, manhattan


def dfs(G: nx.Graph, source: int, destinations: list[int], node_list=None):
  destinations = set(destinations)
  # Run DFS
  nx_edges = nx.dfs_edges(G, source)
  return path_from_nx_edges(nx_edges, source, destinations)


def bfs(G: nx.Graph, source: int, destinations: list[int], node_list=None):
  destinations = set(destinations)
  # Run BFS
  nx_edges = nx.bfs_edges(G, source)
  return path_from_nx_edges(nx_edges, source, destinations)


def gbfs(G: nx.Graph, source: int, destinations: list[int], node_list):
  '''
  Simulate GBFS with A* by setting all edge weights to 0
  '''
  # Avoid mutating the nx.Graph object
  G = G.copy()
  for u, v in G.edges():
    G[u][v]["weight"] = 0

  closest_cost = math.inf
  closest_path = []
  for target in destinations:
    try:
      path = nx.astar_path(G,
                          source=source,
                          target=target,
                          heuristic=lambda n1, n2: manhattan(node_list[n1].coordinates, node_list[n2].coordinates))
      cost = nx.path_weight(G, path)
      if cost < closest_cost:
          closest_cost = cost
          closest_path = path
    except nx.NetworkXNoPath:
      continue

  return closest_path


def shortest_path(G: nx.Graph, source: int, destinations: list[int]):
  destinations = set(destinations)

  # Compute all shortest simple paths from the given source in the graph
  # Returns a dictionary mapping each node to the shortest path to it from source
  lengths, paths = nx.single_source_dijkstra(G, source)

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


reference_methods = {
  "DFS":  dfs,
  "BFS":  bfs,
  "GBFS": gbfs,
  "AS":   shortest_path,
  "CUS1": None,
  "CUS2": shortest_path,
}