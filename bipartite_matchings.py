from collections import defaultdict, deque
from dataclasses import dataclass
from matrix_utils import min_indexes, reduce_each_min
from typing import AbstractSet, Iterable, Mapping, NamedTuple
import numpy as np


@dataclass(eq=True, frozen=True)
class _BranchAndBoundStackFrame:
    graph: dict[int, set[int]]
    matching: dict[int, int]
    matching_index: int


class MinVertexCover(NamedTuple):
    l_minus: AbstractSet[int]
    r_plus: AbstractSet[int]


def enumerate_max_matchings(matrix: np.ndarray) -> Iterable[Mapping[int, int]]:
    workers_count = matrix.shape[0]
    tasks_count = matrix.shape[1]

    init_matching_index = workers_count
    max_matching_index = workers_count + tasks_count

    init_graph = _build_bipartite_graph(matrix)
    init_matching = _find_max_matching(init_graph)
    max_matching_len = len(init_matching)

    init_frame = _BranchAndBoundStackFrame(
        init_graph, init_matching, init_matching_index
    )
    stack = deque([init_frame])
    while stack:
        frame = stack.pop()
        graph = frame.graph

        # Build branch and bound method's node
        matching_index = frame.matching_index
        edge_to = matching_index
        edge_from = frame.matching[edge_to]

        # Right branch. Remove edge from the result matching
        graph_right = graph.copy()
        graph_right[edge_from] = graph_right[edge_from].copy()
        graph_right[edge_from].remove(edge_to)

        matching_right = _find_max_matching(graph_right)
        if len(matching_right) >= max_matching_len:
            # Skip missing vertices in matching_right
            while (
                matching_index not in matching_right
                and matching_index < max_matching_index
            ):
                matching_index += 1

            stack.append(
                _BranchAndBoundStackFrame(graph_right, matching_right, matching_index)
            )

        # Left branch. Include edge to the result matching
        graph_left = graph.copy()
        graph_left[edge_from] = {edge_to}

        # Current edge wasn't removed. frame.matching == _find_max_matching(graph_left)
        matching_left = frame.matching
        if len(matching_left) >= max_matching_len:
            # Move to the next matching_index, current in the answer
            next_matching_index = matching_index + 1

            # Skip missing vertices in matching_left
            while (
                next_matching_index not in matching_left
                and next_matching_index < max_matching_index
            ):
                next_matching_index += 1

            if next_matching_index < max_matching_index:
                stack.append(
                    _BranchAndBoundStackFrame(
                        graph_left, matching_left, next_matching_index
                    )
                )
            else:
                # Return new answer
                yield {
                    left_index: right_index - workers_count
                    for right_index, left_index in matching_left.items()
                }


def find_min_vertex_cover(matrix: np.ndarray) -> MinVertexCover:
    workers_count = matrix.shape[0]

    graph = _build_bipartite_graph(matrix)
    max_matching = _find_max_matching(graph)

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


def _find_max_matching(graph: dict[int, set[int]]) -> dict[int, int]:
    matching = {}
    visited = set()

    def dfs(left_vertex: int) -> bool:
        if left_vertex in visited:
            return False

        visited.add(left_vertex)

        for right_vertex in graph[left_vertex]:
            if right_vertex not in matching or dfs(matching[right_vertex]):
                matching[right_vertex] = left_vertex
                return True

        return False

    for left_vertex in graph:
        visited.clear()
        dfs(left_vertex)

    return matching


def _build_bipartite_graph(matrix: np.ndarray) -> dict[int, set[int]]:
    workers_count = matrix.shape[0]
    reduced_matrix = reduce_each_min(matrix)
    zeros_indexes = min_indexes(reduced_matrix)

    graph = defaultdict(set)
    for row, column in zeros_indexes:
        l_side_index = row
        # Right side indexes start after left side
        r_side_index = workers_count + column

        graph[l_side_index].add(r_side_index)

    return graph
