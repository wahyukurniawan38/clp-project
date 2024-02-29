from math import floor
from typing import Set, List

import numpy as np

from heuristic.block_building.block import Block
from solver.problem import Problem
from solver.utils import get_possible_rotation_mats

def generate_simple_blocks(problem: Problem, max_blocks:int=10000) -> List[Block]:
    possible_rotation_mats = get_possible_rotation_mats()
    block_set_per_type: List[Set[Block]] = [set() for _ in range(len(problem.cargo_type_list))]


    start_idx = 0
    for c_type_idx, cargo_type in enumerate(problem.cargo_type_list):
        if c_type_idx>0:
            start_idx += problem.cargo_type_list[c_type_idx-1].num_cargo
        for rot_mat in possible_rotation_mats:
            real_dim = (cargo_type.dim[np.newaxis,:]*rot_mat).sum(axis=-1)
            num_cargo = cargo_type.num_cargo
            for nx in range(1, num_cargo+1):
                num_cargo_left_1 = int(num_cargo/nx)
                for ny in range(1, num_cargo_left_1+1):
                    num_cargo_left_2 = int(num_cargo/(nx*ny))
                    for nz in range(1, num_cargo_left_2+1):
                        num_used = nx*ny*nz
                        cargo_positions = np.arange(num_used)
                        cargo_positions = np.repeat(cargo_positions[:, np.newaxis], 3, axis=1)
                        cargo_positions[:,0] = cargo_positions[:,0] % nx
                        cargo_positions[:,1] = (cargo_positions[:,1] // nx) % ny
                        cargo_positions[:,2] = (cargo_positions[:,2] // (nx*ny)) % nz
                        cargo_positions = cargo_positions*real_dim[np.newaxis,:]

                        new_block = Block(problem)
                        new_block.dim = np.asanyarray([nx*real_dim[0], ny*real_dim[1], nz*real_dim[2]])
                        new_block.weight = num_used*cargo_type.weight
                        new_block.num_cargo_used[c_type_idx] = num_used
                        new_block.cog = new_block.dim[:2]/2
                        new_block.positions[start_idx:start_idx+num_used,:] =  cargo_positions
                        new_block.rotation_mats[start_idx:start_idx+num_used] = rot_mat
                        block_set_per_type[c_type_idx].add(new_block)
    num_generated_block = sum([len(block_set_per_type[c_type_idx]) for c_type_idx in range(len(problem.cargo_type_list))])
    block_list = []
    if num_generated_block <= max_blocks:
        for block_set in block_set_per_type:
            block_list += list(block_set)
        return block_list
    
    for c_type_idx in range(len(problem.cargo_type_list)):
        block_list_t = list(block_set_per_type[c_type_idx])
        block_list_t = sorted(block_list_t, key= lambda block: block.volume)
        percentage = len(block_list_t)/num_generated_block
        num_block = max(floor(percentage*max_blocks), 1)
        block_list += [block_list_t[:num_block]]
    return block_list        