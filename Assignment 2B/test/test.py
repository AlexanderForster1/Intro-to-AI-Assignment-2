import csv
import sys
import os
import random
import pytest
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from boroondara_search import find_routes
from predictor import predict, predict_single
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
  flow  = round(global_flow_dict[site])
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

@pytest.mark.parametrize("test_id", [f"{i:04d}" for i in range(2, 16)])
def test_consistency(test_id):
  '''Tests that running the same query multiple times returns identical routes and travel time.'''
  route = run_test_case(test_id, max_routes=1)[0]
  path = route["path"]
  time = route["cost_seconds"]
  for i in range(4):
    route = run_test_case(test_id, max_routes=1)[0]
    assert route["path"] == path
    assert route["cost_seconds"] == time

def compare_predictions():
  '''Plots predictions made by different models at random sites across a day'''
  site_ids = random.sample(list(sites.keys()), 12)
  models  = ["lstm", "gru", "rnn"]

  start_date = datetime(2024, 1, 1)
  random_dt  = start_date + timedelta(days=random.randint(0, 365))

  # Create subplot grid
  fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
  axes = axes.flatten()  # FLatten for iteration
  hours = list(range(24))

  for ax, site_id in zip(axes, site_ids):
    preds = defaultdict(list)
    for i in hours:
      # Generate time stamps at each hour of the day
      time = random_dt.replace(hour=i, minute=0, second=0, microsecond=0)
      for model_name in models:
        pred = predict_single(
          scats_number=site_id,
          time=time,
          model_name=model_name,
          time_step=24
        )
        preds[model_name].append(pred)

    # Plot predictions for this site
    for model_name in models:
      ax.plot(hours, preds[model_name], label=model_name)
    ax.set_title(f"SCATS site {site_id:4d}", fontsize=7)
    ax.set_xlabel("Hour", fontsize=7)
    ax.set_ylabel("Flow", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
  
  fig.suptitle(
    f"Traffic predictions on {random_dt.date()}",
    fontsize=12
  )

  # One global legend for the subplots
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper right", fontsize=9)

  plt.tight_layout()
  plt.show()

def compare_predicted_time():
  '''Plots travel time predicted by differet models'''
  test_ids = [f"{i:04d}" for i in range(2, 16)]
  models   = ['lstm', 'gru', 'rnn']
  preds    = defaultdict(list)

  # Predictions based on the exact datetime in each test do not really matter for this comparison so we're just predicting based on the current time to save execution time
  flow_dicts = {
    "lstm": predict(datetime.now(), model_name="lstm"),
    "gru" : predict(datetime.now(), model_name="gru"),
    "rnn" : global_flow_dict,
  }

  for model in models:
    for test_id in test_ids:
      tc = test_cases[test_id]
      # Ignore time in test case
      origin, destination = (
        tc["origin"],
        tc["destination"],
      )
      route = find_routes(
        origin, destination, 
        flow_dict=flow_dicts[model], 
        max_routes=1
      )[0]
      preds[model].append(route["cost_seconds"])
  
  fig, ax = plt.subplots(figsize=(13, 7))
  
  x = np.arange(len(test_ids))
  width = 0.25  # width of the bars

  for i, (model_name, values) in enumerate(preds.items()):
    ax.bar(x + i * width, values, width, label=model_name)

  ax.set_title(f"Predicted Travel Time for Test Cases {test_ids[0]} - {test_ids[-1]}")
  ax.set_xlabel("Test case")
  ax.set_ylabel("Travel time (s)")
  ax.set_xticks(x + width, test_ids)
  ax.legend()

  plt.show()

def compare_weekday_weekend():
  """Plots weekday vs weekend predictions over 24 hours
  for a single week and a single SCATS site."""
  site_id = random.choice(list(sites.keys()))
  models = ["lstm", "gru", "rnn"]
  hours = list(range(24))

  start_of_year = datetime(2007, 1, 1)
  random_week = random.randint(0, 51)
  # Ensure start date is Monday
  monday = start_of_year + timedelta(weeks=random_week)
  monday = monday - timedelta(days=monday.weekday())

  # Create subplot grid
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharey=True)

  for ax, model_name in zip(axes, models):
    weekday_preds = []
    weekend_preds = []

    for hour in hours:
      weekday_hour_preds = []
      weekend_hour_preds = []

      # Sample 4 weeks
      for day_offset in range(28):
        dt = monday + timedelta(days=day_offset)
        time = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        pred = predict_single(
          scats_number=site_id,
          time=time,
          model_name=model_name,
          time_step=24
        )
        # Monday-Friday
        if dt.weekday() < 5:
          weekday_hour_preds.append(pred)
        # Saturday-Sunday
        else:
          weekend_hour_preds.append(pred)

      weekday_preds.append(np.mean(weekday_hour_preds))
      weekend_preds.append(np.mean(weekend_hour_preds))

    ax.plot(hours, weekday_preds, label="Weekday", linewidth=2)
    ax.plot(hours, weekend_preds, label="Weekend", linewidth=2)
    ax.set_title(model_name.upper())
    ax.set_xlabel("Hour of Day")
    ax.grid(alpha=0.3)

  axes[0].set_ylabel("Predicted Traffic Volume")
  fig.suptitle(
    f"Weekday vs Weekend Traffic Predictions\nSCATS Site {site_id}",
    fontsize=14
  )

  # Shared legend
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper right")

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  compare_predictions()
  compare_predicted_time()
  compare_weekday_weekend()