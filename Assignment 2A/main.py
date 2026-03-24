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
node_list = []
current_category = ''
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line in categories:
            current_category = line
        if line != '':
        
            if current_category == 'Nodes:' and line != 'Nodes:':
                node_number = int(line.split(':')[0])
                node_coord = line.split(':')[1].replace('(', '').replace(')', '').split(',')
                node_coord = (int(node_coord[0]), int(node_coord[1]))
                node = Node(node_number, node_coord)
                node_list.insert(node_number, node)
            elif current_category == 'Edges:' and line != 'Edges:':
                edge_info = line.split(':')
                cost = int(edge_info[1].strip())
                nodes = edge_info[0].strip().replace('(', '').replace(')', '').split(',')
                nodes = (int(nodes[0]), int(nodes[1]))
                node_list[nodes[0] - 1].add_edge(nodes[1], cost)
            elif current_category == 'Origin:' and line != 'Origin:':
                origin = int(line.strip())
            elif current_category == 'Destinations:' and line != 'Destinations:':
                destination_nodes = line.strip().split(';')
                destination = [int(destination_node.strip()) for destination_node in destination_nodes]





