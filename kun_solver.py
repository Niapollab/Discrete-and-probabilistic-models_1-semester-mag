from collections import defaultdict, deque
from dataclasses import dataclass
from matrix_utils import min_indexes, reduce_each_min
from typing import Iterable, Iterator
import numpy as np


@dataclass(frozen=True, eq=True)
class _KunStackFrame:
    worker_index: int
    matching: np.ndarray


@dataclass(frozen=True, eq=True)
class _DfsStackFrame:
    vertex_index: int
    neighbours_iter: Iterator[int]


class _KunSolutionIterator(Iterator):
    FIRST_VERTEX = 0
    NO_MATCHING = -1

    _workers_count: int
    _tasks_count: int
    _graph: dict[int, set[int]]
    _kun_stack: deque[_KunStackFrame]

    def __init__(self, matrix: np.ndarray) -> None:
        self._workers_count = matrix.shape[0]
        self._tasks_count = matrix.shape[1]

        if self._workers_count != self._tasks_count:
            raise ValueError('Matrix must be square.')

        self._graph = self.__build_bipartite_graph(matrix)

        init_frame = _KunStackFrame(_KunSolutionIterator.FIRST_VERTEX, np.full(self._tasks_count, _KunSolutionIterator.NO_MATCHING))
        self._kun_stack = deque([init_frame])

    def __next__(self) -> dict[int, int]:
        while self._kun_stack:
            frame = self._kun_stack.pop()

            # Solution was found
            if frame.worker_index >= self._workers_count:
                return _KunSolutionIterator.__build_solution(frame.matching)

            self.__process_frame(frame)

        raise StopIteration

    def __process_frame(self, kun_frame: _KunStackFrame) -> None:
        no_way_needed = False

        vertex = kun_frame.worker_index
        next_vertex = vertex + 1

        vertexes_count = self._workers_count + self._tasks_count
        visited = np.full(vertexes_count, False)

        # Mark first vertrex as visited by default
        visited[vertex] = True

        init_frame = _DfsStackFrame(vertex, iter(self._graph[vertex]))
        dfs_stack = deque([init_frame])

        while dfs_stack:
            dfs_frame = dfs_stack.pop()

            for neighbour_vertex in dfs_frame.neighbours_iter:
                if visited[neighbour_vertex]:
                    continue

                visited[neighbour_vertex] = True
                dfs_stack.append(_DfsStackFrame(neighbour_vertex, iter(self._graph[neighbour_vertex])))

                # If we found not matching vertex
                if kun_frame.matching[neighbour_vertex] == _KunSolutionIterator.NO_MATCHING:
                    no_way_needed = True

                    # Dfs stack contains all path vertex at that moment
                    path = dfs_stack
                    new_matching = _KunSolutionIterator.__build_matching(kun_frame.matching, path)

                    self._kun_stack.append(_KunStackFrame(next_vertex, new_matching))

                    # Remove unnecessary last added stack frame (for micro-optimization purposes)
                    _ = dfs_stack.pop()

                # New dfs stack frame was set, return to the dfs loop
                break

        # Add no way frame, if no kun frames was added previous
        if no_way_needed:
            self._kun_stack.append(_KunStackFrame(next_vertex, kun_frame.matching.copy()))

    def __build_bipartite_graph(self, matrix: np.ndarray) -> dict[int, set[int]]:
        reduced_matrix = reduce_each_min(matrix)
        zeros_indexes = min_indexes(reduced_matrix)

        graph = defaultdict(set)
        for row, column in zeros_indexes:
            l_side_index = row
            # Right side indexes start after left side
            r_side_index = self._workers_count + column

            # Make graph non-oriented
            graph[l_side_index].add(r_side_index)
            graph[r_side_index].add(l_side_index)

        return graph

    @staticmethod
    def __build_solution(matching: np.ndarray) -> dict[int, int]:
        raise NotImplementedError()

    @staticmethod
    def __build_matching(matching: np.ndarray, path: deque[_DfsStackFrame]) -> np.ndarray:
        raise NotImplementedError()


def enumerate_matchings(matrix: np.ndarray) -> Iterable[dict[int, int]]:
    return _KunSolutionIterator(matrix)
