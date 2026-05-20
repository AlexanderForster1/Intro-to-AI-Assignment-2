import csv
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

map_data_path   = Path(__file__).parent.parent / "data" / "map_data.csv"
test_cases_path = Path(__file__).parent / "test_cases.csv"
N = 13  # Number of test cases to generate

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

scats_ids = []
with open(map_data_path, newline="") as f:
  f.readline()  # headers
  for line in f:
    data = line.split(",")
    scats_ids.append(data[0])

start = datetime(2006, 11, 1)
end   = datetime.now()
time_between = end - start

if Path(test_cases_path).exists():
  df = pd.read_csv(test_cases_path)
  id = int(df["test_id"].astype(int).max()) + 1  # get next id
else:
  id = 1

for i in range(id, id+N):
  origin = random.choice(scats_ids)
  while True:
    destination = random.choice(scats_ids)
    if destination != origin:
      break
  
  random_seconds  = random.randint(0, int(time_between.total_seconds()))
  random_datetime = start + timedelta(seconds=random_seconds)
  
  write_to_csv(
    row=[f"{i:04d}", origin, destination, random_datetime],
    filepath=test_cases_path,
    headers=["test_id", "origin", "destination", "datetime"]
  )