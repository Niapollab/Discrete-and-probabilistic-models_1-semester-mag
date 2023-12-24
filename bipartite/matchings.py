from bipartite.graph import build_bipartite_graph
from collections import deque
from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping
import numpy as np


@dataclass(eq=True, frozen=True)
class _BranchAndBoundStackFrame:
    graph: dict[int, set[int]]
    matching: Mapping[int, int]
    matching_index: int


def enumerate_max_matchings(matrix: np.ndarray) -> Iterable[Mapping[int, int]]:
    workers_count = matrix.shape[0]
    tasks_count = matrix.shape[1]

    init_matching_index = workers_count
    max_matching_index = workers_count + tasks_count

    init_graph = build_bipartite_graph(matrix)
    init_matching = _find_max_matching(init_graph)
    max_matching_len = len(init_matching)

    init_frame = _BranchAndBoundStackFrame(
        init_graph, init_matching, init_matching_index
    )
    stack = deque([init_frame])
    while stack:
        frame = stack.pop()
        graph = frame.graph
        matching = frame.matching
        matching_index = frame.matching_index

        # Skip missing vertices in matching
        while (
            matching_index not in matching
            and matching_index < max_matching_index
        ):
            matching_index += 1

        # Return new answer
        if matching_index >= max_matching_index:
            yield {
                left_index: right_index - workers_count
                for right_index, left_index in matching.items()
            }
            continue

        # Build branch and bound method's node
        edge_to = matching_index
        edge_from = matching[edge_to]

        # Right branch. Remove edge from the result matching
        graph_right = graph.copy()
        graph_right[edge_from] = graph_right[edge_from].copy()
        graph_right[edge_from].remove(edge_to)

        matching_right = _find_max_matching(graph_right)
        if len(matching_right) >= max_matching_len:
            stack.append(
                _BranchAndBoundStackFrame(graph_right, matching_right, matching_index)
            )

        # Left branch. Include edge to the result matching
        graph_left = graph.copy()
        graph_left[edge_from] = {edge_to}

        # Current edge wasn't removed. frame.matching == _find_max_matching(graph_left)
        matching_left = matching

        # Move to the next matching_index, current in the answer
        next_matching_index = matching_index + 1
        stack.append(
            _BranchAndBoundStackFrame(
                graph_left, matching_left, next_matching_index
            )
        )


def _find_max_matching(graph: Mapping[int, AbstractSet[int]]) -> Mapping[int, int]:
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
