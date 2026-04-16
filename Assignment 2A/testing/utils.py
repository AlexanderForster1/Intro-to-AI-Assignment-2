from pathlib import Path
from datetime import datetime
import sys
import os
import json
import networkx as nx
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import Node
from dfs import dfs
from bfs import bfs
from cus1 import cus1
from cus2 import cus2

methods = {
  'DFS':  dfs,
  'BFS':  bfs,
  'GBFS': None,
  'AS':   None,
  'CUS1': cus1,
  'CUS2': cus2,
}

def parse_output(input_filepath):
  '''
  Parses the corresponding output file for the given input file.
  Returns a dictionary mapping the method name to the method's output.
  '''
  filename = Path(input_filepath).name
  output_filepath = Path(__file__).resolve().parent / "test_cases" / "outputs" / filename

  output = {}
  current_method = ""

  # Safeguard for undefined output files
  try:
    f = open(output_filepath, "r")
  except FileNotFoundError:
    return None
  
  with open(output_filepath, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      if line in methods:
        output[line] = {}
        current_method = line
        count = 0
        continue
      if count == 0:
        output[current_method]['goal'] = int(line) if line != 'None' else None
      elif count == 1:
        output[current_method]['number_of_nodes'] = int(line)
      elif count == 2:
        output[current_method]['path'] = json.loads(line)
      count += 1
  return output

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

def path_from_nx_edges(nx_edges, source: int, destinations: set[int]) -> list[int]:
  # Record predecessors
  preds: dict[int, int] = {}
  for parent, child in nx_edges:
    preds[child] = parent
    if child in destinations:
      # Reconstruct path
      path = [child]
      while path[-1] != source:
        path.append(preds[path[-1]])
      return list(reversed(path))
  # Edge case: source and destination node is the same
  if source in destinations:
    return [source]
  return []

def path_cost(node_list: list[int, Node], path: list[int]):
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

def manhattan(pos1, pos2):
  x1, y1 = pos1
  x2, y2 = pos2
  return abs(x1-x2) + abs(y1-y2)

def export(node_list: dict[int, Node], origin: int, destinations: list[int]):
  if not node_list:
    return
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  with open(f'test_cases/inputs/g{timestamp}.txt', 'w') as f:
    f.write("Nodes:\n")
    for key, node in node_list.items():
      f.write(f"{key}: ({node.coordinates[0]},{node.coordinates[1]})\n")
    f.write("Edges:\n")
    for key, node in node_list.items():
      for edge in node.edges: 
        f.write(f"({key},{edge[0]}): {edge[1]}\n")
    f.write("Origin:\n")
    f.write(f"{origin if origin is not None else random.choice(list(node_list.keys()))}\n")
    f.write("Destinations:\n")
    if not destinations:
      f.write(f"{random.choice(list(node_list.keys()))}")
    else:
      f.write(f"{"; ".join(str(dest) for dest in destinations)}")