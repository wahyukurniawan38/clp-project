import hashlib

import numpy as np

from heuristic.block_building.utils import init_num_box_used, init_dim, init_packing_area
from solver.problem import Problem
from solver.solution import SolutionBase


class Block(SolutionBase):
    def __init__(self,
                problem: Problem,
                **kwargs) -> None:
        self.problem = problem 
        self.dim = init_dim(kwargs.get("dim"))
        self.num_box_used = init_num_box_used(problem, kwargs.get("num_box_used"))
        self.packing_area = init_packing_area(problem, kwargs.get("packing_area"))
        self.weight = 0 if kwargs.get("weight") is None else kwargs.get("weight")
        self.center_of_gravity = kwargs.get("center_of_gravity")

    # i think the hash of the positions 
    # and the hash of rotation are all we need
    def __hash__(self):
        hashed_pos = hash(self.positions.tobytes())
        hashed_rot_mats = hash(self.rotation_mats.tobytes())
        return hash(hashed_pos+hashed_rot_mats)
    
    def __eq__(self, other: object) -> bool:
        is_position_same = np.allclose(self.positions, other.positions)
        is_rotation_same = np.allclose(self.rotation_mats, other.rotation_mats)
        return is_position_same and is_rotation_same
    
