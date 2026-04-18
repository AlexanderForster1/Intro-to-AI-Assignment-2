import networkx as nx
import sys
import os
import csv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import Node

def createNxGraph(node_list: dict[int, Node]) -> nx.Graph:
  '''
  Builds and returns a NetworkX Graph from node_list.
  Each node in this NetworkX graph is a Node object.
  '''
  g = nx.DiGraph()

  # Add all nodes
  g.add_nodes_from(node_list.keys())

  # Add edges
  # Get a list of all edges in the graph, where each edge is
  # a tuple of (node1, node2, cost)
  nx_edges = []
  for node_number, node in node_list.items():
    node1 = node_number
    for edge in node.edges:
      node2 = edge[0]
      cost = edge[1]
      nx_edges.append((node1, node2, cost))
  g.add_weighted_edges_from(nx_edges)

  return g


def path_cost(node_list: list[int, Node], path: list[int]):
  '''
  Returns the cost of the given path.
  '''
  if not path or len(path) == 1:
    return 0

  total = 0
  for i in range(len(path) - 1):
    current = path[i]
    next    = path[i + 1]

    for neighbour, weight in node_list[current].edges:
      if neighbour == next:
        total += weight
        break

  return total

def shortest_path(G: nx.Graph, source: int, destinations: list[int], node_list=None):
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

def write_to_csv(row, filepath, headers):
  '''
  Write the given row to the CSV file of filepath
  '''
  file_exists = Path(filepath).exists()

  with open(filepath, "a", newline="") as f:
    writer = csv.writer(f)

    # Write header only once
    if not file_exists:
      writer.writerow(headers)

    writer.writerow(row)