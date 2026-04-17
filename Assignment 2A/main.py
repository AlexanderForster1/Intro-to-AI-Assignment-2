# Parse command-line arguments
# import argparse

# parser = argparse.ArgumentParser(description='Search for a path from origin to destination nodes.')
# parser.add_argument('filename', type=str, help='The path to the input file containing nodes, edges, origin, and destinations.')
# parser.add_argument('method', type=str, choices=['BFS', 'DFS', 'AStar', 'GBFS', 'CUS1', 'CUS2'], help='The search method to use: BFS, DFS, AStar, GBFS, CU1, or CU2.')

# args = parser.parse_args()

from gbfs import gbfs

# Represents one graph node with an ID, coordinate tuple, and outgoing edges.
class Node:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        # Each edge is stored as: (neighbor_node_id, edge_cost).
        self.edges = []
        
    def add_edge(self, edge, cost):
        # Add a directed edge from this node to another node.
        self.edges.append((edge, cost))
    

# Input file expected to contain Nodes/Edges/Origin/Destinations sections.
file_path = 'PathFinder-test.txt'
# Section headers used to decide how each line should be parsed.
categories = ['Nodes:', 'Edges:', 'Origin:', 'Destinations:']
# Destination IDs can be a list, while origin is a single node ID.
destination = []
origin = []
# Maps node_id -> Node instance for quick lookup when linking edges.
node_list = {}
current_category = ''
with open(file_path, 'r') as file:
    for line in file:
        # Remove trailing newline/whitespace to simplify checks and splits.
        line = line.strip()
        
        # When we hit a header line, switch parser mode to that section.
        if line in categories:
            current_category = line
            continue
        
        # Ignore blank lines between sections/entries.
        if line == '':
            continue
        
        if current_category == 'Nodes:':
            # Format: "<node_id>:(x,y)"
            node_number = int(line.split(':')[0])
            node_coord = line.split(':')[1].replace('(', '').replace(')', '').split(',')
            node_coord = (int(node_coord[0]), int(node_coord[1]))
            node = Node(node_number, node_coord)
            node_list[node_number] = node
        
        elif current_category == 'Edges:':
            # Format: "(from_node,to_node):cost"
            edge_info = line.split(':')
            cost = int(edge_info[1].strip())
            nodes = edge_info[0].strip().replace('(', '').replace(')', '').split(',')
            nodes = (int(nodes[0]), int(nodes[1]))
            node_list[nodes[0]].add_edge(nodes[1], cost)
        
        elif current_category == 'Origin:':
            # Single origin node ID.
            origin = int(line.strip())
        
        elif current_category == 'Destinations:':
            # One line containing destination IDs separated by ';'.
            destination_nodes = line.strip().split(';')
            destination = [int(destination_node.strip()) for destination_node in destination_nodes]

found_node, total_nodes, path = gbfs(node_list, destination, origin)
print("Visited Nodes:", path, " ", "Number of Nodes Visited:", total_nodes)
if found_node is not None:
    print("Found Destination Node:", found_node)
else:
    print("No destination node found.")

