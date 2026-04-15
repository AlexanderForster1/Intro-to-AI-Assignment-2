import pygame
import sys
import math
import random
import os
from datetime import datetime

'''
CONTROLS

Right-click: Plot a point
Left-click: Select a point (red point = selected)
N: Toggle between node numbers / coordinates
U: Toggle between undirectional - bidirectional edges
E: Export current graph

To draw an edge: Select two points
To set the origin: Select a point + press O
To set the destination: Select a point + press D

Pass the file path as a command-line argument to load an existing test case
(haven't tested this, prolly full of bugs)
e.g. python test_generator.py test_cases/g1.txt

COSTS ARE RANDOMISED but are larger than the Manhattan distance
'''

pygame.init()

PADDING = 40  # Space from edges
WIDTH, HEIGHT = 600 + PADDING * 2, 600 + PADDING * 2
GRID_SIZE = 30

screen = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE     = (255, 255, 255)
GREY      = (200, 200, 200)
DARK_GREY = (120, 120, 120)
BLACK     = (0, 0, 0)
RED       = (255, 0, 0)
BLUE      = (0, 100, 255)

font = pygame.font.SysFont('Consolas', 12)

# Set origins
origin_x = PADDING
origin_y = HEIGHT - PADDING

class Node:
  def __init__(self, name, coordinates):
    self.name = name
    self.coordinates = coordinates
    self.edges = []
      
  def add_edge(self, edge, cost):
    self.edges.append((edge, cost))

node_list = {} 

# plotted[i][j] = k means there is a node k plotted at (i, j)
rows = (WIDTH - PADDING) // GRID_SIZE
plotted = [[-1 for _ in range(rows)] for _ in range(rows)]

origin = None
destinations = []

def parse_input():
  if len(sys.argv) <= 1:
    return
  filename = sys.argv[1]
  categories = ['Nodes:', 'Edges:', 'Origin:', 'Destinations:']
  current_category = ''
  with open(filename, 'r') as file:
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
        plotted[node_coord[0]][node_coord[1]] = node_number
      
      elif current_category == 'Edges:':
        edge_info = line.split(':')
        cost = int(edge_info[1].strip())
        nodes = edge_info[0].strip().replace('(', '').replace(')', '').split(',')
        nodes = (int(nodes[0]), int(nodes[1]))
        node_list[nodes[0]].add_edge(nodes[1], cost)
      
      elif current_category == 'Origin:':
        global origin
        origin = int(line.strip())
      
      elif current_category == 'Destinations:':
        destination_nodes = line.strip().split(';')
        destinations.extend([int(destination_node.strip()) for destination_node in destination_nodes])

def draw_grid():
  # Vertical lines
  x_val = 0
  for x in range(origin_x, WIDTH - PADDING + 10, GRID_SIZE):
    pygame.draw.line(screen, GREY, (x, PADDING), (x, origin_y))
    label = font.render(str(x_val), True, BLACK)
    screen.blit(label, (x - 2, origin_y + 5))
    x_val += 1

  # Horizontal lines
  y_val = 0
  for y in range(origin_y, PADDING - 10, -GRID_SIZE):
    pygame.draw.line(screen, GREY, (origin_x, y), (WIDTH - PADDING, y))
    if y_val == 0:
      y_val = 1 
      continue
    label = font.render(str(y_val), True, BLACK)
    screen.blit(label, (origin_x - 20, y - 5))
    y_val += 1

def draw_axes():
  # X axis
  pygame.draw.line(screen, BLACK, (origin_x, origin_y), (WIDTH - PADDING, origin_y), 2)
  # Y axis
  pygame.draw.line(screen, BLACK, (origin_x, origin_y), (origin_x, PADDING), 2)

def draw_point(x, y, show_numbers, color=BLUE):
  """Plot a point"""
  screen_x = origin_x + x * GRID_SIZE
  screen_y = origin_y - y * GRID_SIZE
  pygame.draw.circle(screen, color, (screen_x, screen_y), 5)

  # label point
  if show_numbers:
    text = str(plotted[x][y])
  else:
    text = f"({x},{y})"
  label = font.render(text, True, BLACK)
  screen.blit(label, (screen_x + 5, screen_y - 5))

def draw_points(show_numbers, selected_node=None):
  """Plot all points"""
  for key, value in node_list.items():
    coords = value.coordinates
    if selected_node is not None and coords == node_list[selected_node].coordinates:
      draw_point(coords[0], coords[1], show_numbers, color=RED)
    else:
      draw_point(coords[0], coords[1], show_numbers)

def draw_edge(pt1, pt2, directed):
  pt1 = grid_to_screen(pt1[0], pt1[1])
  pt2 = grid_to_screen(pt2[0], pt2[1])
  pygame.draw.line(screen, DARK_GREY, pt1, pt2, 1)
  if directed: draw_arrowhead(pt1, pt2)

def draw_edges():
  for key, node in node_list.items():
    for edge in node.edges:
      # Check if it is an directed edge
      directed = True
      for e in node_list[edge[0]].edges:
        if key == e[0]:
          directed = False
          break
      draw_edge(node.coordinates, node_list[edge[0]].coordinates, directed)

def draw_arrowhead(p1, p2):
  arrow_length = 10
  arrow_angle = math.radians(25)

  # Angle of the line
  angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

  # Two sides of the arrow
  left = (
    p2[0] - arrow_length * math.cos(angle - arrow_angle),
    p2[1] - arrow_length * math.sin(angle - arrow_angle),
  )

  right = (
    p2[0] - arrow_length * math.cos(angle + arrow_angle),
    p2[1] - arrow_length * math.sin(angle + arrow_angle),
  )

  pygame.draw.polygon(screen, DARK_GREY, [p2, left, right])

def draw_labels(show_numbers, is_directed):  
  """Draw labels at the top of the window"""  
  if not show_numbers:
    o_text = str(node_list[origin].coordinates) if origin is not None else 'None'
    d_text = f"[{"; ".join([str(node_list[key].coordinates) for key in destinations])}]"
  else:
    o_text = str(origin)
    d_text = f"[{"; ".join(str(dest) for dest in destinations)}]"

  edge_label   = font.render(f"Edge: {"Directed" if is_directed else "Undirected"}", True, BLACK)
  origin_label = font.render("Origin: "+o_text, True, BLACK)
  dest_label   = font.render("Destinations: "+d_text, True, BLACK)

  screen.blit(edge_label, (10, 10))
  screen.blit(origin_label, (180, 10))
  screen.blit(dest_label, (360, 10))

def screen_to_grid(x, y):
  """Convert mouse position to grid coordinates"""
  if x < origin_x or y > origin_y:
    return None
  
  x = round((x - origin_x) / GRID_SIZE)
  y = round((origin_y - y) / GRID_SIZE)
  return (x, y)

def grid_to_screen(x, y):
  """Convert grid coordinates to mouse position"""
  if x < 0 or x > rows or y < 0 or y > rows:
    return None
  
  x = x * GRID_SIZE + origin_x
  y = origin_y - y * GRID_SIZE
  return (x, y)

def is_connected(node1, node2):
  """Returns True if node2 is already in node1's edge list"""
  node = node_list[node1]
  return any(e[0] == node2 for e in node.edges)

def export():
  if not node_list:
    return
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  with open(f'test_cases/inputs/g{timestamp}.txt', 'w') as f:
    f.write("Nodes:\n")
    for key, node in node_list.items():
      f.write(f"{key}: ({node.coordinates[0]},{node.coordinates[1]})\n")
    f.write("Edges:\n")
    for key, node in node_list.items():
      for edge in node.edges: 
        f.write(f"({key},{edge[0]}): {edge[1]}\n")
    f.write("Origin:\n")
    f.write(f"{origin if origin is not None else random.choice(list(node_list.keys()))}\n")
    f.write("Destinations:\n")
    if not destinations:
      f.write(f"{random.choice(list(node_list.keys()))}")
    else:
      f.write(f"{"; ".join(str(dest) for dest in destinations)}")

def main():
  os.system('cls')
  parse_input()

  clock = pygame.time.Clock()
  edging = False  # ;)
  node_name = len(node_list) + 1
  selected_node = None
  show_numbers = False
  is_directed = True

  while True:
    screen.fill(WHITE)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

      if event.type == pygame.MOUSEBUTTONDOWN:
        mouse = pygame.mouse.get_pos()
        grid_pos = screen_to_grid(mouse[0], mouse[1])
        if grid_pos is None:
          continue
        node = plotted[grid_pos[0]][grid_pos[1]]
        # Already plotted, select point instead
        if node != -1:
          # Draw an edge if this is the 2nd point being selected
          if edging and node != selected_node:
            p1 = grid_pos
            p2 = node_list[selected_node].coordinates
            # Edge weight is the Euclidean distance + a random value
            # but this could be changed
            cost = math.ceil(manhattan(p1, p2)) + random.randint(0, 5)
            
            if not is_connected(selected_node, node):
              node_list[selected_node].add_edge(node, cost)

            # Add edge from second node to first node
            if not is_directed and not is_connected(node, selected_node):
              node_list[node].add_edge(selected_node, cost)

            selected_node = None
            edging = False  # :3
          else:
            selected_node = node
            edging = True  
        else:
          node_list[node_name] = Node(node_name, grid_pos)
          plotted[grid_pos[0]][grid_pos[1]] = node_name
          node_name += 1
          selected_node = None
          edging = False

      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_o and selected_node is not None:
          global origin
          origin = selected_node

        if event.key == pygame.K_d and selected_node is not None:
          destinations.append(selected_node)

        selected_node = None
        edging = False
        
        # Toggle between showing node numbers and node coordinates
        if event.key == pygame.K_n:
          show_numbers = not show_numbers

        # Toggle between directed and undirected edges
        if event.key == pygame.K_u:
          is_directed = not is_directed
        
        if event.key == pygame.K_e:
          export()

    draw_grid()
    draw_axes()
    draw_points(show_numbers, selected_node=selected_node)
    draw_edges()
    draw_labels(show_numbers, is_directed)

    pygame.display.flip()
    clock.tick(60)

def manhattan(pos1, pos2):
  x1, y1 = pos1
  x2, y2 = pos2
  return abs(x1-x2) + abs(y1-y2)

if __name__ == "__main__":
  main()