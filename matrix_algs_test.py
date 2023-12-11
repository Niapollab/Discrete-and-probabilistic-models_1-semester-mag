from hungarian_method import enumerate_best_assignments_with_prohibitions
from typing import Sequence
import numpy as np
import pytest


TEST_CASES = [
    (
        # Appointments
        [
            [3, 4, 3, 4, 7, 5],
            [1, 3, 4, 4, 4, 3],
            [5, 3, 7, 6, 4, 5],
            [4, 1, 4, 0, 3, 4],
            [4, 4, 6, 6, 2, 2],
            [5, 5, 3, 1, 2, 2],
        ],
        # Prohibitions
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True]
        ],
    )
]


ANSWERS = [
    (
        [(0, 2), (1, 0), (2, 1), (3, 3), (4, 5), (5, 4)],
        [(0, 2), (1, 0), (2, 1), (3, 3), (4, 4), (5, 5)]
    )
]


@pytest.mark.parametrize(
    'appointments,prohibitions,expected',
    [
        pytest.param(*(np.array(i) for i in input_data), output_data)
        for input_data, output_data in zip(TEST_CASES, ANSWERS)
    ]
)
def test_solution(
    appointments: np.ndarray,
    prohibitions: np.ndarray,
    expected: Sequence[Sequence[tuple[int, int]]]
) -> None:
    actual = [*enumerate_best_assignments_with_prohibitions(appointments, prohibitions)]
    actual = [sorted((int(k), int(v)) for k, v in a.items()) for a in actual]

    assert len(expected) == len(actual)

    for ex_answer in expected:
        assert _compare_answers(ex_answer, actual)


def _compare_answers(
    expected: Sequence[tuple[int, int]],
    actual_answers: Sequence[Sequence[tuple[int, int]]]
) -> bool:
    for actual_answer in actual_answers:
        if actual_answer == expected:
            return True

    return False
