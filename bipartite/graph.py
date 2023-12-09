from bipartite.matrix_utils import min_indexes, reduce_each_min
from collections import defaultdict
import numpy as np


def build_bipartite_graph(matrix: np.ndarray) -> dict[int, set[int]]:
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
