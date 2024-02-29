import pathlib

import numpy as np

from heuristic.block_building.solution import Solution
from heuristic.block_building.generate import generate_simple_blocks
from heuristic.utils import add_container
from solver.problem import read_from_file
from solver.utils import visualize_box


if __name__ == "__main__":
    file_name = "instance_3.json"
    file_path = pathlib.Path()/"instances"/file_name
    problem = read_from_file(file_path.absolute())
    solution = Solution(problem)
    simple_blocks = generate_simple_blocks(problem)
    for block in simple_blocks:
        print(block.dim, block.num_cargo_used)
    print(len(simple_blocks))