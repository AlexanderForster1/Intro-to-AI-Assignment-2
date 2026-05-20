import csv
import sys
import os
import random
import pytest
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from boroondara_search import find_routes
from predict import predict
from graph_builder import _haversine_km, _load_sites
from travel_time import flow_to_speed

test_cases = {}
sites      = _load_sites(Path(__file__).parent.parent / "data" / "map_data.csv")
global_flow_dict = predict(datetime.now(), model_name='rnn')

with open(Path(__file__).parent / "test_cases.csv", "r") as f:
  reader = csv.DictReader(f)

  for row in reader:
    test_cases[row["test_id"]] = {
      "origin": int(row["origin"]),
      "destination": int(row["destination"]),
      "datetime": datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
    }

def run_test_case(
  test_id: str, 
  algorithm: str="AS", 
  max_routes: int=5,
  use_global=False) -> list:
  '''Tests that find_routes returns an empty array if origin and destination are the same.'''
  tc = test_cases[test_id]
  origin, destination, time = (
    tc["origin"],
    tc["destination"],
    tc["datetime"]
  )
  if use_global:
    flow_dict = global_flow_dict
  else:
    flow_dict = predict(time, model_name='rnn')
  return find_routes(origin, destination, flow_dict, algorithm=algorithm, max_routes=max_routes)

def test_same_origin_destination():
  '''Tests that test case 0001 (same origin and destination) returns a path containing a single SCATS site and travelling time of 0'''
  test_id = '0001'
  routes = run_test_case(test_id, use_global=True)
  assert len(routes) == 1
  route = routes[0]
  assert abs(route["cost_minutes"]) <= 1e-9
  assert route["path"] == [test_cases[test_id]["origin"]]

@pytest.mark.parametrize("test_id", [f"{i:04d}" for i in range(2, 16)])
def test_min_travel_time(test_id):
  '''Tests that the estimated travel time for each route is greater than or equal to the minimum possible travel time.'''
  routes = run_test_case(test_id, use_global=True)
  assert len(routes) > 0
  for route in routes:
    travel_time = 0.0
    path = route["path"]

    for j in range(len(path)-1):
      travel_time += 30.0  # seconds per intersection
      site_from = path[j]
      site_to   = path[j+1]
      travel_time += _haversine_km(
        lat1=sites[site_from]["lat"],
        lon1=sites[site_from]["lon"],
        lat2=sites[site_to]["lat"],
        lon2=sites[site_to]["lon"],
      ) / 60.0 * 3600  # hour to second conversion
      assert travel_time <= route["cost_seconds"]

@pytest.mark.parametrize("_", range(5))
def test_flow_to_speed_conversion(_):
  '''Tests that the program accurately converts predicted traffic flow to speed.'''
  site  = random.choice(list(sites.keys()))
  flow  = global_flow_dict[site]
  speed = flow_to_speed(flow)
  if abs(speed - 60.0) <= 1e-9:
    assert flow <= 351
  else:
    assert flow == int(-1.4648375 * (speed ** 2) + 93.75 * speed)

@pytest.mark.parametrize("test_id", [f"{i:04d}" for i in range(2, 16)])
def test_swap(test_id):
  '''Tests that the same route is returned when the origin and destination sites are swapped'''  
  tc = test_cases[test_id]
  origin, destination = (
    tc["origin"],
    tc["destination"],
  )
  route_normal  = find_routes(origin, destination, global_flow_dict, max_routes=1)[0]
  route_swapped = find_routes(destination, origin, global_flow_dict, max_routes=1)[0]
  assert route_normal["path"] == list(reversed(route_swapped["path"]))

def test_predictions():
  # TODO: Compare predictions of different models at one site across a day using charts
  pass

routes = run_test_case("0002")
for route in routes:
  for key, value in route.items():
    print(f"{key}: {value}")