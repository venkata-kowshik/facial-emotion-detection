from collections import deque

graph = {
    'a': ['b', 'c'],
    'b': ['c', 'd'],
    'c': ['a', 'b'],
    'd': ['c']
}

from collections import deque

def bfs(graph, start):
    visited = []        # To keep track of visited nodes
    queue = deque([start])  # Initialize queue with the start node

    while queue:
        node = queue.popleft()  # Remove first element (FIFO)
        if node not in visited:
            print(node, end=' ')
            visited.append(node)
            # Add unvisited neighbors to the queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
bfs(graph, 'a')

