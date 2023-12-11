from sys import stdin, stdout
from typing import Any, Callable, TextIO
import numpy as np
import random as rnd


def read_matrix(
    dtype: Callable[[str], Any] = str, source: TextIO | None = None
) -> np.ndarray:
    source = source or stdin
    rows = []

    while True:
        row = source.readline().strip()

        if not row:
            break

        row_elements = [dtype(element) for element in row.split()]
        rows.append(row_elements)

    if len(rows) > 1 and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('The number of columns must match for all rows.')

    return np.array(rows)


def generate_random_matrix(rows: int, columns: int, min: int, max: int) -> np.ndarray:
    return np.array(
        [[rnd.randint(min, max) for _ in range(columns)] for _ in range(rows)]
    )


def matrix_to_string(
    matrix: np.ndarray,
    element_separator='\t',
    line_separator='\n',
    element_representer: Callable[[Any], str] = lambda element: str(element),
) -> str:
    return line_separator.join(
        element_separator.join(element_representer(element) for element in row)
        for row in matrix
    )


def write_matrix(
    matrix: np.ndarray,
    destination: TextIO = stdout,
    element_separator='\t',
    line_separator='\n',
    element_representer: Callable[[Any], str] = lambda element: str(element),
) -> None:
    destination.write(
        matrix_to_string(matrix, element_separator, line_separator, element_representer)
    )
