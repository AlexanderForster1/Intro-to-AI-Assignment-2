import subprocess
import glob
import pytest
import os
from pathlib import Path

# ANSI colour codes
RED    = "\033[31m"
BLUE   = "\033[1;34m"
GREEN  = "\033[32m"
RESET  = "\033[0m"

methods = {
  "DFS",
  "CUS2"
}

outputs = {
  "DFS": None,
  "BFS": None,
  "GBFS": None,
  "AS": None,
  "CUS1": None,
  "CUS2": None,
}

def parse_output(input_filepath):
  filename = Path(input_filepath).name
  output_filepath = Path(__file__).resolve().parent / "test_cases" / "outputs" / filename

  current_method = ""
  # Safeguard for undefined output files
  try:
    f = open(output_filepath, "r")
  except FileNotFoundError:
    return False
  with open(output_filepath, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      if line in outputs.keys():
        outputs[line] = input_filepath + f" {line}\n"
        current_method = line
        count = 0
        continue
      if count == 0:
        outputs[current_method] += line + " "  # goal
      elif count == 1:
        outputs[current_method] += line + "\n" # number_of_nodes
      else:
        outputs[current_method] += line
      count += 1

  # Debug
  '''
  for key, value in outputs.items():
    print(key)
    print(value + '\n')
  '''
  return True

# Get all input files
input_folder = Path(__file__).resolve().parent / 'test_cases' / 'inputs'
input_filepaths = glob.glob(os.path.join(input_folder, "*.txt"))

# Generate pytest parametrize for all combinations
test_cases = []
for filepath in input_filepaths:
  for method in methods:
    test_cases.append((filepath, method))

@pytest.mark.parametrize("filepath, method", test_cases)
def test_graph_search(filepath, method):
  filename = Path(filepath).name

  # Run main.py
  result = subprocess.run(
    ["python", "main.py", filepath, method],
    capture_output=True,
    text=True
  )

  output = result.stdout.strip()
  errors = result.stderr.strip()

  # Parse expected output
  if not parse_output(filepath):
    pytest.skip(f"Missing output file for {filename}, skipping.")

  expected = outputs.get(method)

  # Logging on failure
  if errors:
    print(f"\n[ERRORS] {filename} [{method}]:\n{errors}")

  # Add -s flag when running this script to see these logs
  print(f"{RED}\n[TEST] {filename} [{method}]{RESET}")
  print(f"{GREEN}[EXPECTED]{RESET}\n{expected}")
  print(f"{GREEN}[ACTUAL]{RESET}\n{output}\n")

  assert output == expected, f"{filename} [{method}] output does not match expected."