from collections import defaultdict, deque
from dataclasses import dataclass
from matrix_utils import min_indexes, reduce_each_min
from typing import Iterable, MutableMapping
import numpy as np


@dataclass(eq=True, frozen=True)
class _BranchAndBoundStackFrame:
    graph: dict[int, set[int]]
    matching: dict[int, int]
    matching_index: int


def enumerate_matchings(matrix: np.ndarray) -> Iterable[MutableMapping[int, int]]:
    init_graph = _build_bipartite_graph(matrix)
    init_matching = _find_max_matching(init_graph)

    max_matching_len = len(init_matching)
    init_frame = _BranchAndBoundStackFrame(init_graph, init_matching, 0)

    stack = deque([init_frame])
    while stack:
        frame = stack.pop()
        graph = frame.graph

        # Build branch and bound method's node
        matching_index = frame.matching_index
        edge_from = matching_index
        edge_to = frame.matching[edge_from]

        # Right branch. Remove edge from the result matching
        graph_right = graph.copy()
        graph_right[edge_from] = graph_right[edge_from].copy()
        graph_right[edge_from].remove(edge_to)

        matching_right = _find_max_matching(graph_right)
        if len(matching_right) >= max_matching_len:
            # Skip missing vertices in matching_right
            while matching_index not in matching_right and matching_index < max_matching_len:
                matching_index += 1

            stack.append(_BranchAndBoundStackFrame(graph_right, matching_right, matching_index))

        # Left branch. Include edge to the result matching
        graph_left = graph.copy()
        graph_left[edge_from] = {edge_to}

        # Current edge wasn't removed. frame.matching == _find_max_matching(graph_left)
        matching_left = frame.matching
        if len(matching_left) >= max_matching_len:
            # Move to the next matching_index, current in the answer
            next_matching_index = matching_index + 1

            # Skip missing vertices in matching_left
            while next_matching_index not in matching_left and next_matching_index < max_matching_len:
                next_matching_index += 1

            if next_matching_index < max_matching_len:
                stack.append(_BranchAndBoundStackFrame(graph_left, matching_left, next_matching_index))
            else:
                # Return new answer
                yield matching_left


def _find_max_matching(graph: dict[int, set[int]]) -> dict[int, int]:
    raise NotImplementedError()


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
