import pathlib

import numpy as np

from solver.utils import visualize_box
from heuristic.insertion.extreme_point.insert import insert_many_cargo_to_one
from heuristic.utils import add_container, insert_cargo_to_container
from solver.problem import read_from_file
from solver.solution import Solution

if __name__ == "__main__":
    file_path = pathlib.Path()/"instances"/"instance_mini_2.txt"
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    solution = add_container(solution, 0)
    solution = insert_many_cargo_to_one(solution, np.asanyarray([0,1,2,3,4,5,6], dtype=int), 0)
    is_inside_container = solution.cargo_container_maps == 0
    visualize_box(solution.container_dims[0], solution.positions[is_inside_container], solution.cargo_dims[is_inside_container], solution.rotation_mats[is_inside_container])
    