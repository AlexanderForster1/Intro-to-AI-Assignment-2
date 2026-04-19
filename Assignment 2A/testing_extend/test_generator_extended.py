import random
import sys
import os
import argparse
from utils import manhattan, export

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import Node


parser = argparse.ArgumentParser(description="Controlled random graph generator.")

parser.add_argument("number_of_nodes", type=int, help="Number of nodes to generate.")
parser.add_argument("number_of_edges", type=int, help="Number of directed edges to generate.")

parser.add_argument("--width", type=int, default=1000, help="Maximum x coordinate.")
parser.add_argument("--height", type=int, default=1000, help="Maximum y coordinate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
parser.add_argument("--bidirectional_prob", type=float, default=0.2, help="Probability an edge is also added in reverse.")
parser.add_argument("--destinations", type=int, default=1, help="Number of destination nodes.")
parser.add_argument("--cost_noise", type=int, default=10, help="Max random extra cost added to Manhattan distance.")
parser.add_argument("--ensure_connected", action="store_true", help="Ensure graph is weakly connected.")
parser.add_argument("--goal_mode", choices=["random", "far", "near"], default="random", help="How to choose destination nodes.")
parser.add_argument("--layout_mode", choices=["uniform", "clustered"], default="uniform", help="How to place nodes.")
parser.add_argument("--disconnect_destinations", action="store_true", help="Disconnect destination nodes from the rest of the graph to create unsolvable cases.")

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

n = args.number_of_nodes
m = args.number_of_edges
max_x = args.width
max_y = args.height

if m < n - 1 and args.ensure_connected:
    raise ValueError("number_of_edges must be at least n-1 when --ensure_connected is used.")

node_list = {}
used_coordinates = set()


def random_coordinate():
    while True:
        x = random.randrange(max_x)
        y = random.randrange(max_y)
        if (x, y) not in used_coordinates:
            used_coordinates.add((x, y))
            return (x, y)


def clustered_coordinate(cluster_centres, cluster_radius=100):
    while True:
        cx, cy = random.choice(cluster_centres)
        x = max(0, min(max_x - 1, cx + random.randint(-cluster_radius, cluster_radius)))
        y = max(0, min(max_y - 1, cy + random.randint(-cluster_radius, cluster_radius)))
        if (x, y) not in used_coordinates:
            used_coordinates.add((x, y))
            return (x, y)


# ----------------------------
# Generate node coordinates
# ----------------------------
cluster_centres = []
if args.layout_mode == "clustered":
    num_clusters = min(5, max(2, n // 20))
    for _ in range(num_clusters):
        cluster_centres.append((random.randrange(max_x), random.randrange(max_y)))

for i in range(1, n + 1):
    if args.layout_mode == "uniform":
        coord = random_coordinate()
    else:
        coord = clustered_coordinate(cluster_centres)

    node_list[i] = Node(i, coord)


def add_edge(i1, i2):
    node1 = node_list[i1]
    node2 = node_list[i2]

    if node1.has_edge(node2):
        return False

    cost = manhattan(node1.coordinates, node2.coordinates) + random.randint(0, args.cost_noise)
    node1.add_edge(i2, cost)
    return True


# ----------------------------
# Ensure base connectivity
# ----------------------------
edge_count = 0

if args.ensure_connected:
    node_ids = list(node_list.keys())
    random.shuffle(node_ids)

    for i in range(1, len(node_ids)):
        a = node_ids[i - 1]
        b = node_ids[i]

        if add_edge(a, b):
            edge_count += 1

        if random.random() < args.bidirectional_prob:
            if add_edge(b, a):
                edge_count += 1


# ----------------------------
# Add remaining random edges
# ----------------------------
max_possible_edges = n * (n - 1)
if m > max_possible_edges:
    raise ValueError(f"Too many edges requested. Maximum possible directed edges is {max_possible_edges}.")

while edge_count < m:
    i1 = random.randint(1, n)
    i2 = random.randint(1, n)

    if i1 == i2:
        continue

    if add_edge(i1, i2):
        edge_count += 1

        if edge_count < m and random.random() < args.bidirectional_prob:
            if add_edge(i2, i1):
                edge_count += 1


# ----------------------------
# Shuffle node order
# ----------------------------
items = list(node_list.items())
random.shuffle(items)
node_list = dict(items)


# ----------------------------
# Choose origin and destinations
# ----------------------------
all_ids = list(node_list.keys())
origin = random.choice(all_ids)

possible_destinations = [node_id for node_id in all_ids if node_id != origin]

if len(possible_destinations) < args.destinations:
    raise ValueError("Not enough nodes available to choose destinations.")

if args.goal_mode == "random":
    destinations = random.sample(possible_destinations, args.destinations)

else:
    origin_coord = node_list[origin].coordinates

    def distance_from_origin(node_id):
        return manhattan(origin_coord, node_list[node_id].coordinates)

    sorted_nodes = sorted(possible_destinations, key=distance_from_origin)

    if args.goal_mode == "near":
        destinations = sorted_nodes[:args.destinations]
    else:  # far
        destinations = sorted_nodes[-args.destinations:]

    if args.disconnect_destinations:
        destination_set = set(destinations)

        for node_id, node in node_list.items():
            if node_id in destination_set:
                # Remove all outgoing edges from destination nodes
                node.edges = []
            else:
                # Remove all incoming edges to destination nodes
                node.edges = [
                    (neighbor, cost)
                    for neighbor, cost in node.edges
                    if neighbor not in destination_set
                ]


export(node_list, origin, destinations)