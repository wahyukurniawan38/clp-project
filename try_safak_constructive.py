import pathlib

import numpy as np

from heuristic.utils import add_container
from heuristic.extreme_point.insertion.insert import insert_many_cargo_to_one
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.lns_safak.insert import add_item_to_container
from heuristic.lns_safak.solution import Solution
from solver.problem import read_from_file
from solver.utils import visualize_box


if __name__ == "__main__":
    file_name = "instance_3.json"
    file_path = pathlib.Path()/"instances"/file_name
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    for i in range(len(problem.container_type_list)):
        for j in range(problem.container_type_list[i].num_available):
            solution = add_container(solution, i)
    cc_idx =  [4,8,9,10]
    cargo_idx = [i for i in range(len(solution.cargo_dims)) if i not in cc_idx]
    cargo_idx = list(range(len(solution.cargo_dims)))
    solution = insert_many_cargo_to_one(solution, np.asanyarray(cc_idx), 0)
    solution = constructive_heuristic(solution)
    print(solution.is_feasible)