import pathlib

import numpy as np

from solver.utils import visualize_box
from heuristic.extreme_point.insertion.insert import insert_many_cargo_to_one
from heuristic.utils import add_container, remove_cargos_from_container, get_unsupported_cargo_idx_from_container
from solver.problem import read_from_file
from solver.solution import SolutionBase

if __name__ == "__main__":
    file_name = "instance_1.json"
    file_path = pathlib.Path()/"instances"/file_name
    problem = read_from_file(file_path.absolute())
    solution = SolutionBase(problem)
    solution = add_container(solution, 0)
    solution = insert_many_cargo_to_one(solution, np.arange(len(solution.cargo_dims)), 0)
    is_inside_container = solution.cargo_container_maps == 0
    visualize_box(solution.container_dims[0], solution.positions[is_inside_container], solution.cargo_dims[is_inside_container], solution.rotation_mats[is_inside_container])
    solution = remove_cargos_from_container(solution, np.asanyarray([0]), 0)
    is_inside_container = solution.cargo_container_maps == 0
    visualize_box(solution.container_dims[0], solution.positions[is_inside_container], solution.cargo_dims[is_inside_container], solution.rotation_mats[is_inside_container])
    cc_unsopperted_idx = get_unsupported_cargo_idx_from_container(solution, 0)
    while len(cc_unsopperted_idx) > 0:
        solution = remove_cargos_from_container(solution, cc_unsopperted_idx, 0)
        cc_unsopperted_idx = get_unsupported_cargo_idx_from_container(solution, 0)
    is_inside_container = solution.cargo_container_maps == 0
    visualize_box(solution.container_dims[0], solution.positions[is_inside_container], solution.cargo_dims[is_inside_container], solution.rotation_mats[is_inside_container])
    