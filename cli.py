from matrix_utils import read_matrix, generate_random_matrix
from typing import Any, Callable, NamedTuple
import numpy as np


class MenuResponse(NamedTuple):
    appointments: np.ndarray
    prohibitions: np.ndarray


def show_menu() -> MenuResponse:
    START_INDEX = 1
    INVALID_MENU_ITEM = 'Unable to recognize menu item.'
    size = len(_MENU)

    while True:
        for i, item in enumerate(_MENU, START_INDEX):
            name, _ = item
            print(f'{i}. {name}')

        menu_item = _read_type('Select menu item: ', INVALID_MENU_ITEM, int)
        if menu_item >= 1 and menu_item < size + START_INDEX:
            _, action = _MENU[menu_item - 1]
            return action()

        print(INVALID_MENU_ITEM)


def _read_from_file() -> MenuResponse:
    INVALID_FILENAME = 'Unable to open the file.'

    appointments = _read_matrix_file('Enter appointments filename: ', INVALID_FILENAME)
    prohibitions = _read_matrix_file('Enter prohibitions filename: ', INVALID_FILENAME).astype(bool)

    return MenuResponse(appointments, prohibitions)


def _generate_random() -> MenuResponse:
    rows_count = _read_type(
        'Enter rows count: ', 'Unable to recognize rows count.', int
    )
    columns_count = _read_type(
        'Enter columns count: ', 'Unable to recognize columns count.', int
    )

    min_value = _read_type('Enter min value: ', 'Unable to recognize min value.', int)
    max_value = _read_type('Enter max value: ', 'Unable to recognize max value.', int)

    appointments = generate_random_matrix(
        rows_count, columns_count, min_value, max_value
    )
    prohibitions = generate_random_matrix(rows_count, columns_count, 0, 1).astype(bool)

    return MenuResponse(appointments, prohibitions)


def _read_matrix_file(prompt: str, error_prompt: str) -> np.ndarray:
    while True:
        try:
            filename = _read_type(prompt, error_prompt)
            with open(filename) as file:
                return read_matrix(int, file)

        except Exception:
            print(error_prompt)


def _read_type(
    prompt: str, error_prompt: str, dtype: Callable[[str], Any] = str
) -> Any:
    while True:
        try:
            value = input(prompt)
            return dtype(value)
        except Exception:
            print(error_prompt)


_MENU = [
    ('Read from file', _read_from_file),
    ('Generate random', _generate_random)
]
