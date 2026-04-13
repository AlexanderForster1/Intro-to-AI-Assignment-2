from input import Node

def dfs(node_list: dict[int, Node], src: int, dest: list[int]):
  '''
  Searches the graph using DFS.
  Returns a tuple of the goal node number, the number of nodes created
  during the search, and the path from the source the goal node.
  '''
  # Check if the source and destination nodes are in the node list
  if node_list.get(src) is None or not(set(dest) & set(node_list)):
    raise ValueError("Source or destination node not in graph")
  
  # explored[k] = True means node k has been explored
  explored = {n: False for n in node_list}

  expanded = {n: False for n in node_list}
  expanded[src] = True

  path = [] 
  result = _dfs(node_list, src, dest, explored, expanded, path)
  # (Goal, number of nodes, path)
  return (result, sum(expanded.values()), path)

# Recursive DFS
def _dfs(node_list, node_number, dest, explored, expanded, path):
  '''
  Performs recursive DFS, returns the destination node if a solution
  is found. Else returns None.
  '''
  explored[node_number] = True
  path.append(node_number)

  # If destination node found
  if node_number in dest:
    return node_number
  
  for edge in node_list[node_number].edges:
    neighbour = edge[0]
    expanded[neighbour] = True
    if not explored[neighbour]:
      result = _dfs(node_list, neighbour, dest, explored, expanded, path)
      if result is not None:
        return result
      
  path.pop()  # Backtrack
  return None