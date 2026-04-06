def dfs(node_list, src, dest):
  # Check if the source and destination nodes are in the node list
  if node_list.get(src) is None or not(set(dest) & set(node_list)):
    raise ValueError("Source or destination node not in graph")
  
  # THIS ASSUMES NODES ARE 1, 2, 3, ... WHICH IS PROBLEMATIC
  # explored[k] = True means node k has been explored
  explored = [False] * (len(node_list) + 1)

  expanded = [False] * (len(node_list) + 1)
  expanded[src] = True

  path = [] 
  result = _dfs(node_list, src, dest, explored, expanded, path)
  # (Goal, number of nodes, path)
  return (result, sum(expanded), path)

# Recursive DFS
def _dfs(node_list, node_number, dest, explored, expanded, path):
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

'''
When all else is equal, nodes should be expanded according to the ascending order, i.e., from the
smaller to the bigger nodes.
-> SORT THE NODES' ADJACENCY LISTS???
'''