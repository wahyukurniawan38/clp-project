import hashlib

import numpy as np

from heuristic.block_building.utils import init_num_cargo_used, init_dim, init_packing_area
from solver.utils import init_positions
from solver.problem import Problem
from solver.solution import SolutionBase

"""
    this is similar to a solution
    it has boxes inside of it
    the boxes' position thus need to be marked
    and so on and so forth, but it does not have
    container.
"""
class Block:
    def __init__(self,
                problem: Problem,
                **kwargs) -> None:
        self.problem = problem 
        self.dim = init_dim(kwargs.get("dim"))
        self.num_cargo_used = init_num_cargo_used(problem, kwargs.get("num_box_used"))
        self.weight = 0 if kwargs.get("weight") is None else kwargs.get("weight")
        self.cog = kwargs.get("cog")
        self.block_position = kwargs.get("block_position")

        # cargo info
        self.cargo_weights = problem.cargo_weights
        self.cargo_dims = problem.cargo_dims
        self.cargo

        # cargo dec variable
        self.positions = init_positions(problem, kwargs.get("positions"))


    # when not considering full support
    # I think the only uniqueness we need about this box
    # is :
        # 1. weight
        # 2. dimension
        # 3. center of gravity
        # 4. number of boxes per box type used
        # 
    def __hash__(self):
        hashed_pos = hash(self.positions.tobytes())
        hashed_rot_mats = hash(self.rotation_mats.tobytes())
        return hash(hashed_pos+hashed_rot_mats)
    
    def __eq__(self, other: object) -> bool:
        is_position_same = np.allclose(self.positions, other.positions)
        is_rotation_same = np.allclose(self.rotation_mats, other.rotation_mats)
        return is_position_same and is_rotation_same
    
