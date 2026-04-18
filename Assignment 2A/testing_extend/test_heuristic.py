import sys
import os
import glob
from pathlib import Path
from utils import shortest_path, path_cost, createNxGraph, write_to_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import parse_input
from gbfs import gbfs
from astar import astar
from cus2 import cus2
from heuristics import heuristics

informed_searches = {
  "GBFS": gbfs,
  "AS"  : astar,
  "CUS2": cus2,
}

# Get all input files
input_folder = Path(__file__).resolve().parent / 'test_cases'
input_filepaths = glob.glob(os.path.join(input_folder, "*.txt"))

# Parse all input files
graphs = {}
nx_graphs = {}
for filepath in input_filepaths:
  node_list, origin, destinations = parse_input(filepath)
  graphs[filepath] = {
    "node_list": node_list,
    "origin": origin,
    "destinations": destinations
  }
  nx_graphs[filepath] = createNxGraph(node_list)

# Loop through all test cases:
for filepath in input_filepaths:
  # Loop through all informed search methods
  for method, search in informed_searches.items():
    # Loop through all heuristic functions
    for h_name, h_fn in heuristics.items():
      filename = Path(filepath).name

      graph  = graphs[filepath]
      result = search(graph["node_list"],
                      graph["origin"],
                      graph["destinations"],
                      heuristic=h_fn)
      
      # If it is a shortest-path algorithm, compare with reference program
      actual_cost    = path_cost(graph["node_list"], result[2])
      if method != "GBFS":
        reference_cost = path_cost(graph["node_list"], shortest_path(nx_graphs[filepath], graph["origin"], graph["destinations"]))
        is_shortest_path = actual_cost == reference_cost
        assert is_shortest_path, f"{filename} [{method}] does not return the shortest path using the {h_name} heuristic function."
      else: is_shortest_path = ""

      # Write results to CSV
      write_to_csv(row=[filename, method,
                        h_name, result[2], actual_cost,
                        is_shortest_path],
                  filepath=Path(__file__).parent / "test_results" / "test_heuristic.csv",
                  headers=["filename", "method", 
                           "heuristic", "path", "path_cost",
                           "is_shortest_path"])