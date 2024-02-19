from typing import List

from cargo.cargo_type import CargoType
from container.container_type import ContainerType
from solver.problem import Problem
from solver.utils import *

class SolutionBase:
    def __init__(self,
                 problem: Problem,
                 **kwargs):
        self.problem = problem

        # from problem objects
        self.cargo_dims = problem.cargo_dims
        self.cargo_types = problem.cargo_types
        self.cargo_weights = problem.cargo_weights
        self.cargo_costs = problem.cargo_costs
        self.cargo_volumes = problem.cargo_volumes

        # init or deepcopy from kwargs
        self.positions = init_positions(problem, kwargs.get("positions"))
        self.cargo_container_maps = init_cargo_container_maps(problem, kwargs.get("cargo_container_maps"))
        self.rotation_mats = init_rotation_mats(problem, kwargs.get("rotation_mats"))
        
        self.nums_container_used = init_nums_container_used(problem, kwargs.get("nums_container_used"))
        self.container_dims = init_container_dims(kwargs.get("container_dims"))
        self.container_max_volumes = init_container_max_volumes(kwargs.get("container_max_volumes"))
        self.container_filled_volumes = init_container_filled_volumes(kwargs.get("contained_filled_volumes"))      
        self.container_max_weights = init_container_max_weights(kwargs.get("container_max_weights"))
        self.container_filled_weights = init_container_filled_weights(kwargs.get("contained_filled_weights"))
        self.container_costs = init_container_costs(kwargs.get("container_costs"))      