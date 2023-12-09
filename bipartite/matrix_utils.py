from typing import Sequence
import numpy as np


def min_indexes(matrix: np.ndarray) -> Sequence[tuple[int, int]]:
    rows, columns = np.where(matrix == matrix.min())
    return [*sorted(zip(rows, columns))]


def reduce_each_min(matrix: np.ndarray) -> np.ndarray:
    return reduce_each_min_column(reduce_each_min_row(matrix))


def reduce_each_min_column(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.min(axis=0)


def reduce_each_min_row(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.min(axis=1)[np.newaxis].T
