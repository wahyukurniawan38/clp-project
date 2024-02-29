import hashlib

import numpy as np

from heuristic.block_building.utils import init_num_cargo_used, init_dim, init_cog, init_block_position
from solver.utils import init_positions, init_rotation_mats
from solver.problem import Problem

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
        self.num_cargo_used = init_num_cargo_used(problem, kwargs.get("num_cargo_used"))
        self.weight = kwargs.get("weight") or 0
        self.volume = kwargs.get("volume") or 0
        self.cog = init_cog(kwargs.get("cog")) 
        self.block_position = init_block_position(kwargs.get("block_position"))

        # cargo info, for brief 
        self.cargo_dims = problem.cargo_dims
        self.cargo_types = problem.cargo_types
        self.cargo_weights = problem.cargo_weights
        self.cargo_costs = problem.cargo_costs
        self.cargo_volumes = problem.cargo_volumes

        # cargo dec variable
        self.positions = init_positions(problem, kwargs.get("positions"))
        self.rotation_mats = init_rotation_mats(problem, kwargs.get("rotation_mats"))
        

    # when not considering full support
    # I think the only uniqueness we need about this box
    # is :
        # 1. number of boxes per box type used
        # 2. dimension
        # 3. center of gravity
    def __hash__(self):
        hashed_dimension = hash(self.dim.tobytes())
        hashed_cog = hash(self.cog.tobytes())
        hashed_num_cargo = hash(self.num_cargo_used.tobytes())
        return hash(hashed_dimension+hashed_cog+hashed_num_cargo)
    
    def __eq__(self, other: object) -> bool:
        is_weight_same = self.weight == other.weight
        is_dimension_same = np.allclose(self.dim, other.dim)
        is_cog_same = np.allclose(self.cog, other.cog)
        is_num_cargo_same = np.all(self.num_cargo_used == other.num_cargo_used)
        return is_weight_same and\
            is_dimension_same and\
            is_cog_same and\
            is_num_cargo_same
    
