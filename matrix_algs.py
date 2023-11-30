from collections import deque
from models import BipartiteGraph
from typing import Sequence


def find_independent_zeros(graph: BipartiteGraph) -> Sequence[tuple[int, int]]:
    def __find_path(curr_vertex: int, visited: list[bool]) -> deque[int] | None:
        visited[curr_vertex] = True

        if curr_vertex == BipartiteGraph.T_VERTEX:
            return deque([curr_vertex])

        for adj_vertex in graph[curr_vertex]:
            if visited[adj_vertex]:
                continue

            path = __find_path(adj_vertex, visited)
            if path is None:
                continue

            path.appendleft(curr_vertex)
            return path

        return None

    for vertex in graph.enumerate_left_side_vertexes():
        visited = [False] * len(graph)
        visited[BipartiteGraph.S_VERTEX] = True

        path = __find_path(vertex, visited)
        if path is None:
            continue

        path.appendleft(BipartiteGraph.S_VERTEX)
        graph.invert_part(path)

    return graph.independent_zeros
