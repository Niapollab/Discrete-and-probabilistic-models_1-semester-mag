from bipartite.graph import build_bipartite_graph
from bipartite.matchings import find_max_matching
from typing import AbstractSet, NamedTuple
import numpy as np


class MinVertexCover(NamedTuple):
    l_minus: AbstractSet[int]
    r_plus: AbstractSet[int]


def find_min_vertex_cover(matrix: np.ndarray) -> MinVertexCover:
    workers_count = matrix.shape[0]

    graph = build_bipartite_graph(matrix)
    max_matching = find_max_matching(graph)

    # If matching is full than return L- only
    left_side_vertices = {*range(workers_count)}
    if len(max_matching) >= workers_count:
        return MinVertexCover(left_side_vertices, set())

    # Inverse matched edges
    for right_index, left_index in max_matching.items():
        graph[left_index].remove(right_index)
        graph[right_index].add(left_index)

    not_in_left_side = left_side_vertices - {*max_matching.values()}
    visited = set()

    def dfs(vertex: int) -> None:
        if vertex in visited:
            return

        visited.add(vertex)

        for neighbour in graph[vertex]:
            dfs(neighbour)

    # Run dfs for every not matched left_vertex
    for left_vertex in not_in_left_side:
        dfs(left_vertex)

    # Add to answer not visited vertex from left side
    l_minus = left_side_vertices - visited

    # Add to answer not visited vertex from right side
    r_plus = {vertex - workers_count for vertex in max_matching.keys() & visited}

    return MinVertexCover(l_minus, r_plus)
