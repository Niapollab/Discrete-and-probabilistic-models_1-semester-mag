from collections import defaultdict
from matrix_utils import min_indexes, reduce_each_min
from typing import AbstractSet, Iterable, Self, Sequence
import copy
import numpy as np


class BipartiteGraph:
    S_VERTEX: int = 0
    T_VERTEX: int = -1

    __vertex_count: int
    __l_side_first: int
    __r_side_first: int
    __adj_list: dict[int, set[int]]

    def __init__(self, matrix: np.ndarray) -> None:
        reduced_matrix = reduce_each_min(matrix)
        self.__vertex_count = reduced_matrix.shape[0]
        if self.__vertex_count < 1:
            raise ValueError("matrix must have at least 1 element.")

        self.__adj_list = defaultdict(set)

        self.__l_side_first = 1
        l_side_last = self.__vertex_count

        self.__r_side_first = self.__vertex_count + 1
        r_side_last = self.__vertex_count * 2

        # Init S vertex
        self.__adj_list[BipartiteGraph.S_VERTEX] = {
            *range(self.__l_side_first, l_side_last + 1)
        }

        # Init L-side
        mins = min_indexes(reduced_matrix)
        for row, column in mins:
            self.__adj_list[self.__l_side_first + row].add(self.__r_side_first + column)

        # Init R-side
        for vertex in range(self.__r_side_first, r_side_last + 1):
            self.__adj_list[vertex].add(BipartiteGraph.T_VERTEX)

        # Init T vertex
        _ = self.__adj_list[BipartiteGraph.T_VERTEX]

    def get_adjacent(self, index: int) -> AbstractSet[int]:
        if index not in self.__adj_list:
            raise ValueError(f'Value "{index}" not in adj_list list.')

        return self.__adj_list[index]

    def invert_part(self, path: Sequence[int]) -> None:
        if len(path) < 2:
            return

        for i in range(len(path) - 1):
            first = path[i]
            second = path[i + 1]

            self.__adj_list[first].remove(second)
            self.__adj_list[second].add(first)

    def enumerate_left_side_vertexes(self) -> Iterable[int]:
        return range(self.__l_side_first, self.__vertex_count + 1)

    def copy(self) -> Self:
        return copy.copy(self)

    @property
    def independent_zeros(self) -> Sequence[tuple[int, int]]:
        result = []

        for i in range(self.__r_side_first, self.__r_side_first + self.__vertex_count):
            row_index = next(iter(self.__adj_list[i]))
            if row_index == BipartiteGraph.T_VERTEX:
                continue

            column_index = i - self.__vertex_count
            result.append((row_index - 1, column_index - 1))

        return result

    def __getitem__(self, index: int) -> AbstractSet[int]:
        return self.get_adjacent(index)

    def __len__(self) -> int:
        return len(self.__adj_list)

    def __copy__(self) -> Self:
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)

        result = copy.copy(obj)
        result.__adj_list = copy.deepcopy(self.__adj_list)

        return result
