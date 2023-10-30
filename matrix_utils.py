from random import sample
from typing import Any, Callable, Dict

import numpy as np

transpose = np.transpose


def read_matrix(new_line_provider: Callable[[], str], element_parser: Callable[[str], Any] = lambda element: element):
    rows = []

    while True:
        row = new_line_provider().strip()

        if not row:
            break

        row_elements = [element_parser(element) for element in row.split()]
        rows.append(row_elements)

    if len(rows) > 1 and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('The number of columns must match for all rows.')

    return np.array(rows)


def read_matrix_from_cli(message: str, element_parser: Callable[[str], Any] = lambda element: element) -> np.ndarray:
    if message is not None:
        print(message, end='')

    return read_matrix(lambda: input(), element_parser)


def read_bool_matrix_from_cli(message: str) -> np.ndarray:
    return read_matrix_from_cli(message, lambda element: element != '0')


def read_int_matrix_from_cli(message: str) -> np.ndarray:
    return read_matrix_from_cli(message, lambda element: int(element))


def read_float_matrix_from_cli(message: str) -> np.ndarray:
    return read_matrix_from_cli(message, lambda element: float(element))


def read_matrix_from_file(filename: str, element_parser: Callable[[str], Any] = lambda element: element) -> np.ndarray:
    with open(filename, 'r') as file:
        return read_matrix(lambda: file.readline(), element_parser)


def read_bool_matrix_from_file(filename: str) -> np.ndarray:
    return read_matrix_from_file(filename, lambda element: element != '0')


def read_int_matrix_from_file(filename: str) -> np.ndarray:
    return read_matrix_from_file(filename, lambda element: int(element))


def read_float_matrix_from_file(filename: str) -> np.ndarray:
    return read_matrix_from_file(filename, lambda element: float(element))


def matrix_to_string(matrix: np.ndarray, element_separator = '\t', line_separator = '\n', element_representer: Callable[[Any], str] = lambda element: str(element)) -> str:
    return line_separator.join(element_separator.join(element_representer(element) for element in row) for row in matrix)


def adjacency_matrix_to_string(matrix: np.ndarray, element_separator = '\t', line_separator = '\n') -> str:
    return matrix_to_string(matrix, element_separator, line_separator, lambda element: '1' if element else '0')


def generate_random_adjacency_matrix(vertex: int, arcs: int) -> np.ndarray:
    arr = np.full(vertex * vertex, False)

    for i in sample([*range(0, arr.shape[0])], arcs):
        arr[i] = True

    return arr.reshape((vertex, vertex))


def adjacency_list_to_matrix(adjacency_list: Dict[int, set[int]]) -> np.ndarray:
    adjacency_list_len = len(adjacency_list)
    matrix = np.full((adjacency_list_len, adjacency_list_len), False)

    vertex_mapper = {name: index
        for index, name in enumerate(adjacency_list.keys())}

    for adjacency_vertexes in adjacency_list:
        for vertex in adjacency_list[adjacency_vertexes]:
            if vertex not in vertex_mapper:
                raise ValueError(f'Unknown adjacency vertex name "{vertex}".')

            row = vertex_mapper[adjacency_vertexes]
            column = vertex_mapper[vertex]
            matrix[row][column] = True

    return matrix
