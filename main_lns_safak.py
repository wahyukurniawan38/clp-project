import pathlib

import numpy as np

from heuristic.utils import add_container
from heuristic.extreme_point.insertion.insert import insert_many_cargo_to_one
from heuristic.lns_safak.insert import add_item_to_container
from heuristic.lns_safak.solution import Solution
from solver.problem import read_from_file
from solver.utils import visualize_box


if __name__ == "__main__":
    file_name = "instance_2.json"
    file_path = pathlib.Path()/"instances"/file_name
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    solution = add_container(solution, 0)
    # cc_idx =  []
    # cargo_idx = [i for i in range(len(solution.cargo_dims)) if i not in cc_idx]
    cargo_idx = list(range(len(solution.cargo_dims)))
    # solution = insert_many_cargo_to_one(solution, np.asanyarray(cc_idx), 0)
    solution, not_inserted_idx = add_item_to_container(solution, np.asanyarray(cargo_idx), 0, "layer-building")
    is_inside_container = solution.cargo_container_maps == 0
    visualize_box(solution.container_dims[0], solution.positions[is_inside_container], solution.cargo_dims[is_inside_container], solution.rotation_mats[is_inside_container])
    
    print(not_inserted_idx)