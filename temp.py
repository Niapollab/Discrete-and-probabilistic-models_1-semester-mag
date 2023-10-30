from matrix_algs import (build_reachability_matrix, colorize, find_core,
                         find_external_stability_set,
                         find_internal_stability_set, find_maximum_matchings,
                         find_strong_components,
                         to_bipartite_transport_network)
from matrix_utils import (adjacency_list_to_matrix, adjacency_matrix_to_string,
                          read_bool_matrix_from_file, transpose)

matrix = read_bool_matrix_from_file('matrix.txt')
print('Матрица смежности:', adjacency_matrix_to_string(matrix), sep='\n')

reachability_matrix = build_reachability_matrix(matrix)
print('Матрица достижимости:', adjacency_matrix_to_string(reachability_matrix), sep='\n')

counterreachability_matrix = transpose(reachability_matrix)
print('Матрица контрдостижимости:', adjacency_matrix_to_string(counterreachability_matrix), sep='\n')

strong_components = find_strong_components(reachability_matrix & counterreachability_matrix)
print(f'Сильные компоненты:', '\n'.join(f'{index + 1}: {sorted(element)}' for index, element in enumerate(strong_components)), sep='\n')

internal_stability_set = find_internal_stability_set(matrix)
print('Множества внутренней устойчивости:', '\n'.join(f'{index + 1}: {sorted(element)}' for index, element in enumerate(internal_stability_set)), sep='\n')

external_stability_set = find_external_stability_set(matrix)
print('Множества внешней устойчивости:', '\n'.join(f'{index + 1}: {sorted(element)}' for index, element in enumerate(external_stability_set)), sep='\n')

core = find_core(internal_stability_set, external_stability_set)
print('Ядро:', '\n'.join(f'{index + 1}: {sorted(element)}' for index, element in enumerate(core)), sep='\n')
