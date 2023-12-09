from bipartite.graph import build_bipartite_graph
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np


@dataclass(eq=True, frozen=True)
class _BranchAndBoundStackFrame:
    graph: dict[int, set[int]]
    matching: dict[int, int]
    matching_index: int


def enumerate_max_matchings(matrix: np.ndarray) -> Iterable[Mapping[int, int]]:
    workers_count = matrix.shape[0]
    tasks_count = matrix.shape[1]

    init_matching_index = workers_count
    max_matching_index = workers_count + tasks_count

    init_graph = build_bipartite_graph(matrix)
    init_matching = find_max_matching(init_graph)
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

        matching_right = find_max_matching(graph_right)
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


def find_max_matching(graph: dict[int, set[int]]) -> dict[int, int]:
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
