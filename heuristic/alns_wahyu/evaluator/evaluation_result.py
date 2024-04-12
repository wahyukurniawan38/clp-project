from typing import List

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.objective import compute_obj
from solver.solution import SolutionBase

class EvaluationResult:
    def __init__(self, 
                 df_cargos:pd.DataFrame,
                 df_containers:pd.DataFrame,
                 x: np.ndarray, 
                 solution_list:List[SolutionBase],
                 omega:float=0.99):
        self.x = x
        self.df_cargos = df_cargos
        self.df_containters = df_containers
        self.solution_list = solution_list
        self.omega = omega

    @property
    def is_feasible(self):
        return self.is_all_cargo_packed and self.is_all_cog_feasible

    @property
    def score(self):
        return compute_obj(self.x, self.df_cargos, self.df_containters, self.omega)

    @property
    def positions(self):
        _, num_cargo = self.x.shape 
        cargo_positions = np.empty((num_cargo,2), float)
        i = 0
        for ccm in self.x:
            if not np.any(ccm):
                continue
            cargo_positions[ccm,:] = self.solution_list[i].positions
            i+=1
        return cargo_positions
    
    @property
    def real_cargo_dims(self):
        _, num_cargo = self.x.shape 
        real_cargo_dims = np.empty((num_cargo,3), float)
        i = 0
        for ccm in self.x:
            if not np.any(ccm):
                continue
            real_cargo_dims[ccm,:] = self.solution_list[i].real_cargo_dims
            i+=1
        return real_cargo_dims
    
    @property
    def rotation_mats(self):
        _, num_cargo = self.x.shape 
        rotation_mats = np.empty((num_cargo,3,3), float)
        i = 0
        for ccm in self.x:
            if not np.any(ccm):
                continue
            rotation_mats[ccm,:,:] = self.solution_list[i].rotation_mats
            i+=1
        return rotation_mats

    @property
    def is_all_cargo_packed(self):
        for solution in self.solution_list:
            if solution is None:
                continue
            if not np.all(solution.cargo_container_maps >=0):
                return False
        return True
    
    @property
    def is_all_cog_feasible(self):
        for solution in self.solution_list:
            if solution is None:
                continue
            if not solution.is_cog_feasible:
                return False
            print(solution.container_cogs)
        return True