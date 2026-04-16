from utils import parse_output, createNxGraph, path_cost, methods
from pathlib import Path
from reference import reference_methods
import glob
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import parse_input

# ------------------------- SETUP ------------------------- #

# Get all input files
input_folder = Path(__file__).resolve().parent / 'test_cases' / 'inputs'
input_filepaths = glob.glob(os.path.join(input_folder, "*.txt"))

graphs    = {}  # graphs[filepath] contains node list, origin, and destinations parsed from filepath
nx_graphs = {}  # nx_graphs[filepath] contains the NetworkX Graph version of graphs[filepath]
outputs   = {}  # outputs[filepath] contains expected output from the corresponding output file

for filepath in input_filepaths:
  node_list, origin, destinations = parse_input(filepath)
  graphs[filepath] = {
    "node_list": node_list,
    "origin": origin,
    "destinations": destinations
  }
  nx_graphs[filepath] = createNxGraph(node_list)
  output = parse_output(filepath)
  if output is not None:
    outputs[filepath] = output

# Generate pytest parametrize for all combinations
test_cases = []
for filepath in input_filepaths:
  for method in methods:
    test_cases.append((filepath, method))

# ------------------------- TEST 1 ------------------------- #

@pytest.mark.parametrize("filepath, method", test_cases[:(7*len(methods))])
def test_complete(filepath, method):
  '''
  Tests the full output of each search method, including the goal, number of nodes, and path.
  Only applies to test cases where expected outputs had been manually calculated.
  g01.txt -> g07.txt
  '''
  if methods[method] is None:
    pytest.skip("Skipping due to unimplemented search method.")
    
  filename = Path(filepath).name

  graph  = graphs[filepath]
  output = outputs[filepath][method]

  if not output:
    pytest.skip("Skipping due to lack of expected output.")

  result = methods[method](graph["node_list"], graph["origin"], graph["destinations"])
  assert result == (output['goal'], output['number_of_nodes'], output['path']), f"{filename} [{method}] output does not match expected."

# ------------------------- TEST 2 ------------------------- #

@pytest.mark.parametrize("filepath, method", test_cases)
def test_valid_path(filepath, method):
  '''
  Tests that the path returned by each search method is a valid path, i.e. the path exists and is traversable in the given graph.
  '''
  if methods[method] is None:
    pytest.skip("Skipping due to unimplemented search method.")
  filename = Path(filepath).name
  graph    = graphs[filepath]
  result   = methods[method](graph["node_list"], graph["origin"], graph["destinations"])

  node_list = graph['node_list']
  path      = result[2]
  for i in range(len(path)-1):
    current = path[i]
    next    = path[i+1]
    if (next not in (e[0] for e in node_list[current].edges)):
      assert False, f"[{method}] does not return a valid path in {filename}."
  assert True

# ------------------------- TEST 3 ------------------------- #

@pytest.mark.parametrize("filepath, method", test_cases)
def test_path_match(filepath, method):
  '''
  Tests that the path returned by each search method matches the path returned by the reference program. If the two paths do not match, check that the costs of the paths are the same, as the reference program might expand nodes in a different order than the implemented program.
  '''
  if methods[method] is None:
    pytest.skip("Skipping due to unimplemented search method.")
  if reference_methods[method] is None:
    pytest.skip("Skipping due to unavailable reference method.")

  filename = Path(filepath).name
  graph    = graphs[filepath]
  nx_graph = nx_graphs[filepath]
  shortest_paths = ["AS", "CUS2"]

  reference_path = reference_methods[method](nx_graph, graph["origin"], graph["destinations"])
  result         = methods[method](graph["node_list"], graph["origin"], graph["destinations"])
  actual_path    = result[2]

  reference_cost = path_cost(graph["node_list"], reference_path)
  actual_cost    = path_cost(graph["node_list"], actual_path)

  # Check that the actual path matches the reference path
  # If they don't, check if the two paths at least have the same cost. As the implemented algorithms might expand nodes in a different order than NetworkX, the output paths might be different despite having the same cost. If that is the case, check that the actual path contain nodes of smaller indexes than the reference path.

  # Case 1: Exact match
  if actual_path == reference_path:
    assert True

  # Case 2: Different paths but same cost: only applies to shortest paths algorithms
  elif actual_cost == reference_cost and method in shortest_paths:
    assert actual_path < reference_path, \
      f"{filename} [{method}] path differs from reference and does not expand node in ascending order when all else is equal."
    
  # Case 3: Straight up wrong
  else:
    assert False, \
      f"{filename} [{method}] path does not match reference."