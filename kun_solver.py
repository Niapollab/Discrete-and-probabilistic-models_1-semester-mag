from collections import defaultdict, deque
from dataclasses import dataclass
from matrix_utils import min_indexes, reduce_each_min
from typing import Iterable, Iterator, Sequence
import numpy as np


@dataclass(frozen=True, eq=True)
class _KunStackFrame:
    worker_index: int
    matching: np.ndarray


@dataclass(frozen=True, eq=True)
class _DfsStackFrame:
    left_index: int
    right_index: int
    right_neighbours_iter: Iterator[int]


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

        vertexes_count = self._workers_count + self._tasks_count
        init_frame = _KunStackFrame(_KunSolutionIterator.FIRST_VERTEX, np.full(vertexes_count, _KunSolutionIterator.NO_MATCHING))
        self._kun_stack = deque([init_frame])

    def __next__(self) -> dict[int, int]:
        while self._kun_stack:
            frame = self._kun_stack.pop()

            # Solution was found
            if frame.worker_index >= self._workers_count:
                return self.__build_solution(frame.matching)

            self.__process_frame(frame)

        raise StopIteration

    def __process_frame(self, kun_frame: _KunStackFrame) -> None:
        no_way_needed = True

        init_vertex = kun_frame.worker_index
        next_vertex = init_vertex + 1

        vertexes_count = kun_frame.matching.shape[0]
        visited = np.full(vertexes_count, False)

        # Mark first vertrex as visited by default
        visited[init_vertex] = True

        init_frame = _DfsStackFrame(init_vertex, _KunSolutionIterator.NO_MATCHING, iter(self._graph[init_vertex]))
        dfs_stack = deque([init_frame])

        while dfs_stack:
            dfs_frame = dfs_stack[-1]

            for neighbour_vertex in dfs_frame.right_neighbours_iter:
                if visited[neighbour_vertex]:
                    continue

                # Select next vertex (from left side) for dfs continuation
                left_index = kun_frame.matching[neighbour_vertex]
                right_index = neighbour_vertex

                visited[neighbour_vertex] = True
                if left_index != _KunSolutionIterator.NO_MATCHING:
                    # Enumerate neighbours for left-side vertex (all neighbours will be right-sided)
                    right_neighbours_iter = iter(self._graph[left_index])

                    # Set new dfs stack frame and return to the dfs loop
                    dfs_stack.append(_DfsStackFrame(left_index, right_index, right_neighbours_iter))
                    break

                # We found not matching vertex
                no_way_needed = False

                # Dfs stack contains all path vertex at that moment
                path = [*((frame.left_index, frame.right_index) for frame in dfs_stack), (left_index, right_index)]
                new_matching = _KunSolutionIterator.__build_matching(kun_frame.matching, path)

                self._kun_stack.append(_KunStackFrame(next_vertex, new_matching))
            else:
                # Remove dfs stack frame if all neighbours was viewed
                _ = dfs_stack.pop()

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

            graph[l_side_index].add(r_side_index)

        return graph

    def __build_solution(self, matching: np.ndarray) -> dict[int, int]:
        vertexes_count = self._workers_count + self._tasks_count

        # Answer start from self._workers_count in matching
        answer = {}
        for i in range(self._workers_count, vertexes_count):
            left_index = matching[i]
            right_index = i - self._workers_count

            answer[left_index] = right_index

        return answer

    @staticmethod
    def __build_matching(matching: np.ndarray, path: Sequence[tuple[int, int]]) -> np.ndarray:
        new_matching = matching.copy()

        # Start building from the ending of path [path_length - 1:0)
        for i in range(len(path) - 1, 0, -1):
            from_vertex, _ =  path[i - 1]
            _, to_vertex = path[i]

            new_matching[to_vertex] = from_vertex

        return new_matching


def enumerate_matchings(matrix: np.ndarray) -> Iterable[dict[int, int]]:
    return _KunSolutionIterator(matrix)
