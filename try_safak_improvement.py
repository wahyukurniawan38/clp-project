import pathlib

import numpy as np

from heuristic.utils import add_container
from heuristic.extreme_point.insertion.insert import insert_many_cargo_to_one
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.lns_safak.improvement import improvement_heuristic
from heuristic.lns_safak.insert import add_item_to_container
from heuristic.lns_safak.solution import Solution, create_copy
from solver.problem import read_from_file
from solver.utils import visualize_box


if __name__ == "__main__":
    file_name = "n40_0_n20_2_nc_40_u_0.9.json"
    file_path = pathlib.Path()/"instances"/file_name
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    for i in range(len(problem.container_type_list)):
        for j in range(problem.container_type_list[i].num_available):
            solution = add_container(solution, i)
    solution = constructive_heuristic(solution)
    g = 0.2
    ld = 0.2
    max_perturb_iter = 1000
    max_repair_iter = 10
    best_solution = create_copy(solution)
    solution, best_solution = improvement_heuristic(solution, best_solution, g,ld, max_perturb_iter, max_repair_iter)
    for i in range(len(best_solution.container_dims)):
        is_inside_container = best_solution.cargo_container_maps == i
        visualize_box(best_solution.container_dims[i], best_solution.positions[is_inside_container], best_solution.cargo_dims[is_inside_container], best_solution.rotation_mats[is_inside_container])
    