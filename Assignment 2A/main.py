# Parse command-line arguments
import argparse
from dfs import dfs
from cus2 import cus2

methods = {
    'BFS':  None,
    'DFS':  dfs,
    'AS':   None,
    'GBFS': None,
    'CUS1': None,
    'CUS2': cus2,
}
parser = argparse.ArgumentParser(description='Search for a path from origin to destination nodes.')
parser.add_argument('filename', type=str, help='The path to the input file containing nodes, edges, origin, and destinations.')
parser.add_argument('method', type=str, choices=methods.keys(), 
                    help=f'The search method to use: {", ".join(methods.keys())}')

args = parser.parse_args()

class Node:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self.edges = []
        
    def add_edge(self, edge, cost):
        self.edges.append((edge, cost))
    
file_path = args.filename

categories = ['Nodes:', 'Edges:', 'Origin:', 'Destinations:']
destination = []
origin = []
node_list = {}
current_category = ''
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        
        if line in categories:
            current_category = line
            continue
        
        if line == '':
            continue
        
        if current_category == 'Nodes:':
            node_number = int(line.split(':')[0])
            node_coord = line.split(':')[1].replace('(', '').replace(')', '').split(',')
            node_coord = (int(node_coord[0]), int(node_coord[1]))
            node = Node(node_number, node_coord)
            node_list[node_number] = node
        
        elif current_category == 'Edges:':
            edge_info = line.split(':')
            cost = int(edge_info[1].strip())
            nodes = edge_info[0].strip().replace('(', '').replace(')', '').split(',')
            nodes = (int(nodes[0]), int(nodes[1]))
            node_list[nodes[0]].add_edge(nodes[1], cost)
        
        elif current_category == 'Origin:':
            origin = int(line.strip())
        
        elif current_category == 'Destinations:':
            destination_nodes = line.strip().split(';')
            destination = [int(destination_node.strip()) for destination_node in destination_nodes]

# Sort the adjacency list of each node so that nodes are expanded in ascending order 
# (NOTE 1 in asm specifications)
for number, node in node_list.items():
    node.edges.sort(key=lambda e: e[0])

result = methods[args.method](node_list, origin, destination)
print(result)  # Temp output