from typing import Set

from heuristic.block_building.block import Block
from solver.problem import Problem
from solver.utils import get_possible_rotation_mats

def generate_simple_blocks(problem: Problem, max_blocks:int=10000):
    possible_rotation_mats = get_possible_rotation_mats()
    blocks: Set[Block] = set()

    for cargo_type in problem.cargo_type_list:
        for rot_mat in possible_rotation_mats:
            real_dim = cargo_type.dim