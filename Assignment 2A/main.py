# Parse command-line arguments
#import argparse

#parser = argparse.ArgumentParser(description='Search for a path from origin to destination nodes.')
#parser.add_argument('filename', type=str, help='The path to the input file containing nodes, edges, origin, and destinations.')
#parser.add_argument('method', type=str, choices=['bfs', 'dfs', 'astar', 'gbfs', 'cu1', 'cu2'], help='The search method to use: bfs, dfs, astar, gbfs, cu1, or cu2.')

#args = parser.parse_args()

class Node:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self.edges = []
        
    def add_edge(self, edge, cost):
        self.edges.append((edge, cost))
    

file_path = 'PathFinder-test.txt'
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




