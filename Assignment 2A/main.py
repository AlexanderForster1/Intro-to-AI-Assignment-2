# Parse command-line arguments
import argparse
from input import parse_input
from dfs import dfs
from bfs import bfs
from astar import astar
from cus1 import cus1
from cus2 import cus2

methods = {
    'BFS':  bfs,
    'DFS':  dfs,
    'AS':   astar,
    'GBFS': None,
    'CUS1': cus1,
    'CUS2': cus2,
}

parser = argparse.ArgumentParser(description='Search for a path from origin to destination nodes.')
parser.add_argument('filename', type=str, help='The path to the input file containing nodes, edges, origin, and destinations.')
parser.add_argument('method', type=str, choices=methods.keys(), 
                    help=f'The search method to use: {", ".join(methods.keys())}')

args = parser.parse_args()

node_list, origin, destination = parse_input(args.filename)
result = methods[args.method](node_list, origin, destination)
goal, number_of_nodes, path = result
output = f'''{args.filename} {args.method}
{goal} {number_of_nodes}
{path}'''
print(output)
