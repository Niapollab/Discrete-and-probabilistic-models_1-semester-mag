from matrix_algs import find_independent_zeros
from models import BipartiteGraph
from typing import Sequence
import numpy as np
import pytest


TEST_CASES = [
    (
        # Appointments
        [[1, 0, 2, 0],
        [0, 2, 3, 0],
        [0, 1, 0, 2],
        [1, 2, 0, 0]],

        # Prohibitions
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]
    )
]


ANSWERS = [
    (
        # Independent zeros
        [
            [(1, 0), (0, 1), (2, 2), (3, 3)],
            [(1, 3), (0, 1), (2, 0), (3, 2)]
        ]
    )
]

@pytest.mark.parametrize('appointments,prohibitions,expected', [pytest.param(*(np.array(i) for i in input_data), output_data) for input_data, output_data in zip(TEST_CASES, ANSWERS)])
def test_check_independent_zeros(appointments: np.ndarray, prohibitions: np.ndarray, expected: Sequence[tuple[int, int]]) -> None:
    graph = BipartiteGraph(appointments)

    actual = find_independent_zeros(graph)

    for answer in expected:
        if answer == actual:
            return

    assert False
