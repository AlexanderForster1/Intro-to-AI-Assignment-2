import random
import sys
import os
import argparse
from utils import manhattan, export

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from input import Node

parser = argparse.ArgumentParser(description='Randomly generate an input text file for Assignment 2A.')
parser.add_argument('number_of_nodes', type=int, help='The number of nodes to be generated.')
parser.add_argument('number_of_edges', type=int, help='The number of edges to be generated')

args = parser.parse_args()

max_x, max_y = (1000, 1000)             # Limits of the Cartensian plane
n            = args.number_of_nodes     # n nodes
m            = args.number_of_edges     # m edges

node_list = {}
# plotted[i][j] = True means (i, j) is already plotted
plotted = [[False for _ in range(max_y+1)] for _ in range(max_x+1)]

# Plot random N points
for i in range(1, n+1):
  while True:
    rand_x, rand_y = (random.randrange(max_x), random.randrange(max_y))
    if not plotted[rand_x][rand_y]:
      break
  node         = Node(i, (rand_x, rand_y))
  node_list[i] = node

# Plot random m edges
i = 0
while i < m:
  # Selet two random distinct points
  i1 = random.randint(1, n)
  while True:
    i2 = random.randint(1, n)
    if i1 != i2: break

  node1 = node_list[i1]
  node2 = node_list[i2]
  
  # Edge weight is Manhattan distance + random value to keep heuristic admissible
  random_cost = manhattan(node1.coordinates, node2.coordinates) + random.randint(0, 10)

  if not node1.has_edge(node2):
    node1.add_edge(i2, random_cost)
    i += 1
  # 20% of the time it is a bidirectional edge
  if random.random() < 0.2 and not node2.has_edge(node1):
    node_list[i2].add_edge(i1, random_cost)
    i += 1

# Shuffle
items = list(node_list.items())
random.shuffle(items)
node_list = dict(items)

# Select origin
origin = random.choice(list(node_list.keys()))

# Select destinations
destinations = set()
iters = random.randint(1, 5)
for i in range(iters):
  destinations.add(random.choice(list(node_list.keys())))
destinations = list(destinations)

export(node_list, origin, destinations)