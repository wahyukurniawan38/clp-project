import pathlib

import numpy as np

from heuristic.insertion.extreme_point.insert import insert_many_cargo_to_one
from heuristic.utils import add_container, insert_cargo_to_container
from solver.problem import read_from_file
from solver.solution import Solution

if __name__ == "__main__":
    file_path = pathlib.Path()/"instances"/"instance_mini_2.txt"
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    solution = add_container(solution, 0)
    solution = insert_cargo_to_container(solution, 3, 0, np.eye(3,3), np.asanyarray([0,0,0]))
    solution = insert_cargo_to_container(solution, 0, 0, np.eye(3,3), np.asanyarray([0,1,0]))
    solution = insert_cargo_to_container(solution, 2, 0, np.eye(3,3), np.asanyarray([0,2,0]))
    solution = insert_cargo_to_container(solution, 1, 0, np.eye(3,3), np.asanyarray([0,0,1]))
    solution = insert_many_cargo_to_one(solution, np.asanyarray([4,5,6], dtype=int), 0)
    