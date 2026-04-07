import subprocess
import sys
import glob
import os
from pathlib import Path

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

def test_search():
  # Get all input files
  input_folder = Path(__file__).resolve().parent / 'test_cases' / 'inputs'
  input_filepaths = glob.glob(os.path.join(input_folder, "*.txt"))

  for filepath in input_filepaths:
    for method in methods:
      # Run the main script
      result = subprocess.run(
        ["python", "main.py", filepath, method],
        capture_output=True,        
        text=True  # Return output as string
      )

      # Get the printed output
      '''
      filename method
      goal number_of_nodes
      path
      '''
      output = result.stdout.strip()
      errors = result.stderr.strip()
      if errors: print(errors)
      if not parse_output(filepath):
        continue

      assert output == outputs[method]