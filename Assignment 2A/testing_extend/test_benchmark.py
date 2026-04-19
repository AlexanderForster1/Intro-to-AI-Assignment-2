import sys
import os
import glob
import time
import threading
import psutil
from pathlib import Path
from utils import path_cost, createNxGraph, write_to_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import parse_input
from gbfs import gbfs
from astar import astar
from cus2 import cus2
from bfs import bfs
from dfs import dfs
from cus1 import cus1


methods = {
    'BFS': bfs,
    'DFS': dfs,
    'AS': astar,
    'GBFS': gbfs,
    'CUS1': cus1,
    'CUS2': cus2,
}


class ResourceMonitor:
    def __init__(self, pid=None, interval=0.01):
        self.process = psutil.Process(pid or os.getpid())
        self.interval = interval
        self.running = False
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.thread = None

    def _monitor(self):
        # Prime CPU percent readings
        self.process.cpu_percent(interval=None)

        while self.running:
            try:
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent(interval=None)

                if mem_mb > self.peak_memory_mb:
                    self.peak_memory_mb = mem_mb

                if cpu_percent > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu_percent

                time.sleep(self.interval)
            except psutil.Error:
                break

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()


def benchmark_search(search_func, node_list, origin, destinations):
    monitor = ResourceMonitor()

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    monitor.start()
    result = search_func(node_list, origin, destinations)
    monitor.stop()

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    runtime_ms = (end_wall - start_wall)*1000  # Convert to milliseconds
    cpu_time_ms = (end_cpu - start_cpu)*1000  # Convert to milliseconds

    return result, runtime_ms, cpu_time_ms, monitor.peak_memory_mb, monitor.peak_cpu_percent

# Get all input files
input_folder = Path(__file__).resolve().parent / 'test_cases'
input_filepaths = glob.glob(os.path.join(input_folder, "*.txt"))

# Parse all input files once
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

# Output CSV
csv_path = Path(__file__).parent / "test_results" / "test_benchmark.csv"

# Loop through all test cases and methods
for filepath in input_filepaths:
    filename = Path(filepath).name
    graph = graphs[filepath]

    for method_name, search_func in methods.items():
        if method_name == "CUS2" and filename in ["g21.txt", "g28.txt"]:
            print(f"Skipping {filename} for CUS2 as it is too slow.")
            continue
        
        for i in range(3):  # Run each method 3 times for averaging

            result, runtime_ms, cpu_time_ms, peak_memory_mb, peak_cpu_percent = benchmark_search(
                search_func,
                graph["node_list"],
                graph["origin"],
                graph["destinations"]
            )

            # Assumes result = (goal, node_count, path)
            goal = result[0] if result is not None and len(result) > 0 else None
            node_count = result[1] if result is not None and len(result) > 1 else None
            path = result[2] if result is not None and len(result) > 2 else None

            actual_cost = path_cost(graph["node_list"], path) if path else None
            path_length = len(path) if path else 0
            goal_found = goal is not None

            print(f"Completed {filename} [{method_name}] run {i+1}/3")
            write_to_csv(
                row=[
                    filename,
                    method_name,
                    i+1,
                    actual_cost,
                    runtime_ms,
                    cpu_time_ms,
                    peak_memory_mb,
                    peak_cpu_percent,
                    node_count,
                    path_length,
                    goal_found
                ],
                filepath=csv_path,
                headers=[
                    "filename",
                    "method",
                    "run_index",
                    "path_cost",
                    "runtime_ms",
                    "cpu_time_ms",
                    "peak_memory_mb",
                    "peak_cpu_percent",
                    "nodes_expanded",
                    "path_length",
                    "goal_found"
                ]
        )