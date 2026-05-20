import csv
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from boroondara_search import find_routes
from predict import predict

test_cases = {}

with open(Path(__file__).parent / "test_cases.csv", "r") as f:
  reader = csv.DictReader(f)

  for row in reader:
    test_cases[row["test_id"]] = {
      "origin": row["origin"],
      "destination": row["destination"],
      "datetime": datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
    }

def run_test_case(test_id: str) -> list:
  '''Tests that find_routes returns an empty array if origin and destination are the same.'''
  tc = test_cases[test_id]
  origin, destination, time = (
    tc["origin"],
    tc["destination"],
    tc["datetime"]
  )
  flow_dict = predict(time=time)
  return find_routes(origin, destination, flow_dict)

def test_same_origin_destination():
  assert run_test_case("0001") == []

routes = run_test_case("0002")
for route in routes:
  for key, value in route.items():
    print(f"{key}: {value}")