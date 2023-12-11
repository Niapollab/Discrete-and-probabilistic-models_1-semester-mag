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


def enumerate_best_assignments_with_prohibitions(appointments: np.ndarray, prohibitions: np.ndarray, logger: logging.Logger | None = None) -> Iterable[Mapping[int, int]]:
    _validate_prohibitions(prohibitions)
    logger = logger or logging.getLogger('dummy')

    appointments_workers_count = appointments.shape[0]
    prohibitions_workers_count = prohibitions.shape[0]
    if appointments_workers_count != prohibitions_workers_count:
        raise ValueError('Appointments workers count must be equals to prohibitions.')

    appointments_tasks_count = appointments.shape[1]
    prohibitions_tasks_count = prohibitions.shape[1]
    if appointments_tasks_count != prohibitions_tasks_count:
        raise ValueError('Appointments tasks count must be equals to prohibitions.')

    logger.info('Appointments matrix:\n%s', Lazy(lambda: matrix_to_string(appointments)))
    logger.info('Prohibitions matrix:\n%s', Lazy(lambda: matrix_to_string(prohibitions.astype(int))))

    logger.info('Apply prohibitions to appointments')
    max_fine = appointments.max() + 1
    new_appointments = np.where(prohibitions, appointments, max_fine)

    for solution in enumerate_best_assignments(new_appointments, logger):
        yield solution


def enumerate_best_assignments(
    appointments: np.ndarray, logger: logging.Logger | None = None
) -> Iterable[Mapping[int, int]]:
    _validate_appointments(appointments)

    workers_count = appointments.shape[0]
    tasks_count = appointments.shape[1]

    # Add dummy tasks
    logger = logger or logging.getLogger('dummy')
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
        logger.debug('Solution:\n%s', _log_solution(solution))

        for solution in solutions:
            solution = _remove_dummies(solution, dummies)
            yield solution
            logger.debug('Solution:\n%s', _log_solution(solution))

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


def _validate_appointments(appointments: np.ndarray) -> None:
    if appointments.dtype != int:
        raise ValueError('Appointments matrix must be int matrix.')

    if (appointments < 0).any():
        raise ValueError('Appointments must be positive matrix.')

    workers_count = appointments.shape[0]
    tasks_count = appointments.shape[1]

    if tasks_count > workers_count:
        raise ValueError('Task count must be less or equals to workers count.')


def _validate_prohibitions(prohibitions: np.ndarray) -> None:
    if prohibitions.dtype != bool:
        raise ValueError('Prohibitions matrix must be bool matrix.')

    workers_availability = prohibitions.sum(axis=0)
    if not workers_availability.all():
        raise ValueError('Some workers unavailable.')

    tasks_availability = prohibitions.sum(axis=1)
    if not tasks_availability.all():
        raise ValueError('Some tasks unavailable.')

    workers_count = prohibitions.shape[0]
    tasks_count = prohibitions.shape[1]

    if tasks_count > workers_count:
        raise ValueError('Task count must be less or equals to workers count.')
