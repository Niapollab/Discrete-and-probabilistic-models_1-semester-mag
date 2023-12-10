from bipartite import (
    reduce_each_min,
    enumerate_max_matchings,
    find_min_vertex_cover,
    reduce_by_min_vertex_cover
)
from lazy import Lazy, LazyCounter
from matrix_utils import matrix_to_string
from typing import Iterable, Mapping
import logging
import numpy as np


def enumerate_best_assignments(
    appointments: np.ndarray, logger: logging.Logger | None = None
) -> Iterable[Mapping[int, int]]:
    workers_count = appointments.shape[0]
    tasks_count = appointments.shape[1]

    if tasks_count > workers_count:
        raise ValueError('Task count must be less or equals to workers count.')

    logger = logger or logging.getLogger('dummy')

    # Add dummy tasks
    dummies = {*range(tasks_count, workers_count)}
    if dummies:
        logger.info('Current matrix:\n%s', Lazy(lambda: matrix_to_string(appointments)))
        logger.warning(
            'Task count %d less than workers count %d. Matrix will be squared by zero columns.',
            tasks_count,
            workers_count
        )

        shape = (workers_count, len(dummies))
        zeros = np.zeros(shape, dtype=int)
        appointments = np.append(appointments, zeros, axis=1)

    iteration = LazyCounter(1)
    while True:
        logger.info('Current iteration: %s', iteration)
        logger.info('Current matrix:\n%s', Lazy(lambda: matrix_to_string(appointments)))

        appointments = reduce_each_min(appointments)
        logger.info('Reduced matrix:\n%s', Lazy(lambda: matrix_to_string(appointments)))

        solutions = iter(enumerate_max_matchings(appointments))

        first_solution = next(solutions, None)
        if not first_solution:
            logger.error('Unable to find solutions for this matrix')
            break

        solution = first_solution
        logger.info('Suggested solution:\n%s', _log_solution(solution))

        if len(first_solution) < workers_count:
            logger.warning(
                'Solution has length %d, but expected %d. Reducing by minimum vertex cover be applied',
                len(solution),
                workers_count
            )

            appointments = _reduce_matrix(appointments, logger)
            continue

        # Return answers
        logger.info('Suggested solution is full matched')

        solution = _remove_dummies(solution, dummies)
        yield solution
        logger.info('Solution:\n%s', _log_solution(solution))

        for solution in solutions:
            solution = _remove_dummies(solution, dummies)
            yield solution
            logger.info('Solution:\n%s', _log_solution(solution))

        break


def _reduce_matrix(matrix: np.ndarray, logger: logging.Logger) -> np.ndarray:
    min_vertex_cover = find_min_vertex_cover(matrix)

    l_minus_lazy = Lazy(
        lambda: f'L-: {', '.join(str(vertex + 1) for vertex in min_vertex_cover.l_minus)}'
    )
    r_plus_lazy = Lazy(
        lambda: f'R+: {', '.join(str(vertex + 1) for vertex in min_vertex_cover.r_plus)}'
    )
    logger.info(
        'Minimum vertex cover:\n%s',
        Lazy(lambda: '\n'.join(str(lazy) for lazy in (l_minus_lazy, r_plus_lazy)))
    )

    return reduce_by_min_vertex_cover(matrix, min_vertex_cover)


def _log_solution(solution: Mapping[int, int]) -> Lazy:
    return Lazy(
        lambda: '\n'.join(
            f'{left + 1} -> {right + 1}' for left, right in solution.items()
        )
    )


def _remove_dummies(
    solution: Mapping[int, int], dummies: set[int]
) -> Mapping[int, int]:
    if not dummies:
        return solution

    return {
        left_index: right_index
        for left_index, right_index in solution.items()
        if left_index not in dummies
    }
