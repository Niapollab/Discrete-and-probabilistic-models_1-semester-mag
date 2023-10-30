from typing import Any, Dict, Iterable, Sequence

import numpy as np


def build_reachability_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    res = adjacency_matrix.copy()

    for k in range(res.shape[0]):
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = res[i][j] | (res[i][k] & res[k][j])

    return res


def find_strong_components(strong_components_matrix: np.ndarray) -> np.ndarray:
    return np.array([s
        for s in [set(i + 1 for i in range(unique_row.shape[0]) if unique_row[i])
            for unique_row in np.unique(strong_components_matrix, axis=0)]
        if s])


def find_internal_stability_set(adjacency_matrix: np.ndarray) -> np.ndarray:
    disjunctors = ([{i + 1}, {j + 1}]
        for i in range(adjacency_matrix.shape[0])
        for j in range(i + 1, adjacency_matrix.shape[1])
        if adjacency_matrix[i][j] | adjacency_matrix[j][i])

    disjunctive_normal_form = __find_disjunctive_normal_form(disjunctors)

    minimum_disjunctive_normal_form = __find_minimum_disjunctive_normal_form(disjunctive_normal_form)

    ALL_VERTEX_SET = {*range(1, adjacency_matrix.shape[0] + 1)}
    return np.array([ALL_VERTEX_SET - element
        for element in minimum_disjunctive_normal_form])


def find_external_stability_set(adjacency_matrix: np.ndarray) -> np.ndarray:
    disjunctors = ([{j + 1}
            for j in range(adjacency_matrix.shape[1])
            if i == j or adjacency_matrix[i][j]]
        for i in range(adjacency_matrix.shape[0]))

    disjunctive_normal_form = __find_disjunctive_normal_form(disjunctors)

    minimum_disjunctive_normal_form = __find_minimum_disjunctive_normal_form(disjunctive_normal_form)

    return minimum_disjunctive_normal_form


def find_core(internal_stability_set: np.ndarray, external_stability_set: np.ndarray) -> np.ndarray:
    return np.array([first
        for first in internal_stability_set
        for second in external_stability_set
        if __sets_is_equals(first, second)])


def colorize(adjacency_matrix: np.ndarray, colors: int) -> np.ndarray:
    def _has_edge(first: int, second: int) -> bool:
        return adjacency_matrix[first][second] or adjacency_matrix[second][first]

    def _dfs(vertex_to: int, vertex_colors: np.ndarray) -> None:
        available_colors = {*range(colors)}

        for vertex_from in range(adjacency_matrix.shape[0]):
            if vertex_from == vertex_to or not _has_edge(vertex_from, vertex_to) or vertex_colors[vertex_from] < 0:
                continue

            available_colors.discard(vertex_colors[vertex_from])

        if not available_colors:
            raise ValueError(f'It is not possible to color a graph with {colors} colors.')

        first_available = next(iter(available_colors))
        vertex_colors[vertex_to] = first_available

        for vertex in range(adjacency_matrix.shape[0]):
            if vertex == vertex_to or not _has_edge(vertex_to, vertex) or vertex_colors[vertex] >= 0:
                continue

            _dfs(vertex, vertex_colors)

    vertex_colors = np.full(adjacency_matrix.shape[0], -1)

    for vertex_to in range(adjacency_matrix.shape[0]):
        if vertex_colors[vertex_to] != -1:
            continue

        _dfs(vertex_to, vertex_colors)

    return vertex_colors


def to_bipartite_transport_network(adjacency_matrix: np.ndarray) -> np.ndarray:
    colors = colorize(adjacency_matrix, 2)

    adjacency_matrix = np.insert(adjacency_matrix, 0, [element == 0 for element in colors], axis=0)
    adjacency_matrix = np.insert(adjacency_matrix, adjacency_matrix.shape[0], [0] * adjacency_matrix.shape[1], axis=0)
    adjacency_matrix = np.insert(adjacency_matrix, adjacency_matrix.shape[1], [0, *colors, 0], axis=1)
    adjacency_matrix = np.insert(adjacency_matrix, 0, [0] * adjacency_matrix.shape[0], axis=1)

    return adjacency_matrix


def find_maximum_matchings(bipartite_transport_network_adjacency_matrix: np.ndarray) -> Dict[int, int]:
    def _dfs(vertex: int, px: np.ndarray, py: np.ndarray, visited: np.ndarray) -> bool:
        if visited[vertex]:
            return False

        visited[vertex] = True

        for adj_vertex in range(bipartite_transport_network_adjacency_matrix.shape[1]):
            if not bipartite_transport_network_adjacency_matrix[vertex][adj_vertex]:
                continue

            if py[adj_vertex] == -1:
                py[adj_vertex] = vertex
                px[vertex] = adj_vertex
                return True
            else:
                if _dfs(py[adj_vertex], px, py, visited):
                    py[adj_vertex] = vertex
                    px[vertex] = adj_vertex
                    return True

        return False

    px = np.full(bipartite_transport_network_adjacency_matrix.shape[0], -1)
    py = np.full(bipartite_transport_network_adjacency_matrix.shape[1], -1)

    is_path = True
    while is_path:
        is_path = False
        visited = np.full(bipartite_transport_network_adjacency_matrix.shape[0], False)
        for vertex in range(bipartite_transport_network_adjacency_matrix.shape[0]):
            if not bipartite_transport_network_adjacency_matrix[0][vertex]:
                continue

            if px[vertex] == -1:
                is_path |= _dfs(vertex, px, py, visited)

    result = {x: px[x]
        for x in range(bipartite_transport_network_adjacency_matrix.shape[0])
        if bipartite_transport_network_adjacency_matrix[0][x] and bipartite_transport_network_adjacency_matrix[px[x]][-1]}

    return result


def __find_minimum_disjunctive_normal_form(disjunctive_normal_form: np.ndarray) -> np.ndarray:
    required_sets = np.full(disjunctive_normal_form.shape, True)

    for i in range(disjunctive_normal_form.shape[0]):
        for j in range(disjunctive_normal_form.shape[0]):
            if i == j or not required_sets[i] or not required_sets[j]:
                continue

            if len(disjunctive_normal_form[i] - disjunctive_normal_form[j]) == 0:
                required_sets[j] = False

    return np.array([dnm for dnm, rs in zip(disjunctive_normal_form, required_sets) if rs])


def __find_disjunctive_normal_form(disjunctors: Iterable[Sequence[set[int]]]) -> np.ndarray:
    disjunctive_normal_form = next(iter(disjunctors), None)
    if disjunctive_normal_form is None:
        return np.empty(0)

    for disjunctor in disjunctors:
        disjunctive_normal_form = [f | s
            for f in disjunctive_normal_form
            for s in disjunctor]

    return __find_unique_sets(disjunctive_normal_form)


def __find_unique_sets(list_of_sets: Sequence[set[Any]]) -> np.ndarray:
    unique_sets = []

    for s in list_of_sets:
        is_duplicate = any(__sets_is_equals(s, unique_set)
            for unique_set in unique_sets)

        if not is_duplicate:
            unique_sets.append(s)

    return np.array(unique_sets)


def __sets_is_equals(first: set[Any], second: set[Any]) -> bool:
    return len(first - second) == 0 and len(second - first) == 0
