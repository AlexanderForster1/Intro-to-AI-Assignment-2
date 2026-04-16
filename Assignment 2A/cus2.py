from input import Node
from heuristics import manhattan_distance
import math
import heapq
import sys

# Increase recursion depth limit
sys.setrecursionlimit(10**5)

def cus2(node_list: dict[int, Node], src: int, dest: list[int]):
  '''
  Searches the graph using recursive best first search (RBFS).
  Returns a tuple of the goal node number, the number of nodes created
  during the search, and the path from the source the goal node.
  '''
  # Check if the source and destination nodes are in the node list
  if node_list.get(src) is None or not(set(dest) & set(node_list)):
    raise ValueError("Source or destination node not in graph")
  
  h0 = min(manhattan_distance(node_list[src], node_list[d]) for d in dest)
  src_node = (src, h0)  # (node number, f value)
  visited = {n: False for n in node_list}

  # (Goal, number of nodes, path)
  path, _ = _rbfs(node_list, src_node, dest, math.inf, 0, [src], visited)
  goal = path[-1] if path else None
  return (goal, sum(visited.values()), path)

# node = (node number, node's f value)
def _rbfs(node_list, node, dest, f_limit, node_g, path, visited):
  '''
  Performs recursive RBFS. Returns an empty array or a one-item array
  containing the destination node and its f-value.
  '''
  node_number, node_f = node
  visited[node_number] = True

  if node_number in dest:
    return ([node_number], node_f)  # Destination node found
  
  # Compute f-values for all successors
  pq = []

  successors = node_list[node_number].edges  
  g = {}
  for edge in successors:
    neighbour = edge[0]
    if neighbour in path:
      continue

    g[neighbour] = node_g + edge[1]  # Cost from src node to successor node

    h = min(manhattan_distance(node_list[neighbour], node_list[d]) for d in dest)
    # As backtracking only happens if the current becomes worse than the alternative
    f = max(g[neighbour] + h, node_f)
    heapq.heappush(pq, (f, neighbour))  # (node's f value, node number)

  if not pq:
    return [], math.inf

  while True:
    # Explore the most promising node
    best = heapq.heappop(pq)
    best_node = (best[1], best[0])  # (node number, node f)  

    # Backtrack if the current path exceeds the best alternative
    if best[0] > f_limit:
      return ([], best[0])
    
    # Keep track of the next best path
    alternative_f = pq[0][0] if pq else math.inf  # Second best f value
    new_limit = min(f_limit, alternative_f)
    # When the current path is better than the alternative, keep searching deeper
    result, best_f = _rbfs(node_list, 
                           best_node, dest, 
                           new_limit, 
                           g[best[1]], 
                           path + [best[1]],
                           visited
                           )

    # If destination node found, return result
    # else explore the next best successor
    if result:
      return [node_number] + result, best_f
    
    # If subtree is exhausted, do not reinsert
    if best_f == math.inf:
      # If there are no candidates left, stop exploring this subtree completely
      if not pq:
        return [], math.inf
      # Else try the other candidates
      else:
        continue
    
    # Push the explored node back onto the heap with its new f-value
    heapq.heappush(pq, (best_f, best[1]))