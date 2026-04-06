import math
import heapq

def cus2(node_list, src, dest):
  # Check if the source and destination nodes are in the node list
  if node_list.get(src) is None or not(set(dest) & set(node_list)):
    raise ValueError("Source or destination node not in graph")
  
  h0 = min(manhattan(node_list, src, d) for d in dest)
  src_node = (src, h0)  # (node number, f value)
  expanded = {n: False for n in node_list.keys()}
  expanded[src] = True

  # (Goal, number of nodes, path)
  path, _ = _rbfs(node_list, src_node, dest, math.inf, 0, [src], expanded)
  goal = path[-1] if path else None
  return (goal, sum(expanded.values()), path)

# node = (node number, node's f value)
def _rbfs(node_list, node, dest, f_limit, node_g, path, expanded):
  node_number, node_f = node

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
    expanded[neighbour] = True

    g[neighbour] = node_g + edge[1]  # Cost from src node to successor node

    h = min(manhattan(node_list, neighbour, d) for d in dest)
    # As backtracking only happens if the current becomes worse than the alternative
    f = max(g[neighbour] + h, node_f)
    heapq.heappush(pq, (f, neighbour))  # (node's f value, node number)

  if not pq:
    return [], math.inf

  while True:
    best = heapq.heappop(pq)
    best_node = (best[1], best[0])  # (node number, node f)  

    # Backtrack if the current path exceeds the best alternative
    if best[0] > f_limit:
      return ([], best[0])
    
    alternative_f = pq[0][0] if pq else math.inf  # Second best f value
    new_limit = min(f_limit, alternative_f)
    # If the current path is better than the alternative, keep searching deeper
    result, best_f = _rbfs(node_list, 
                           best_node, dest, 
                           new_limit, 
                           g[best[1]], 
                           path+[best[1]],
                           expanded
                           )

    # Push the explored node back onto the heap with its new f-value
    heapq.heappush(pq, (best_f, best[1]))

    # If destination node found, return result
    # else explore the next best successor
    if result:
      return [node_number] + result, best_f

# TEMPORARY HEURISTIC CODE TO TEST THAT IT WORKS
# REMOVE AND REPLACE WITH SHARED HEURISTIC CODE
def manhattan(node_list, node, dest):
  node_coord = node_list[node].coordinates
  dest_coord = node_list[dest].coordinates
  return abs(node_coord[0] - dest_coord[0]) + abs(node_coord[1] - dest_coord[1])