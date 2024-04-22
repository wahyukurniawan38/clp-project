from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd


from heuristic.alns_wahyu.objective import compute_obj
from solver.solution import SolutionBase
from solver.utils import visualize_box

class EvaluationResult:
    def __init__(self, 
                 df_cargos:pd.DataFrame,
                 df_containers:pd.DataFrame,
                 x: np.ndarray,
                 solution_list:List[SolutionBase],
                 omega:float=0.99,
                 cargo_loads:np.array=None):
        self.x = x
        self.df_cargos = df_cargos
        self.df_containters = df_containers
        self.cargo_weights = df_cargos["weight"].to_numpy(copy=False)
        self.cargo_volumes = df_cargos["vol"].to_numpy(copy=False)
        self.cargo_loads = cargo_loads
        if self.cargo_loads is None:
            self.cargo_loads = self.cargo_weights*self.cargo_volumes
        self.cargo_ratios = df_cargos["ratio"].to_numpy(copy=False)
        self.cargo_prices = df_cargos["price"].to_numpy(copy=False)
        self.container_cost = df_containers["e"].to_numpy(copy=False)[0]
        self.container_dim =  np.array([df_containers['length'][0], df_containers['width'][0], df_containers['height'][0]])
        self.max_container_volume = df_containers["volume"].to_numpy(copy=False)[0]
        self.max_container_weight = df_containers["weight"].to_numpy(copy=False)[0]
        self.solution_list: List[SolutionBase] = solution_list
        self.omega = omega

    def get_container_fig(self, ct_idx):
        container_dim = self.container_dim
        solution = self.solution_list[ct_idx]
        if len(solution.cargo_dims)==0:
            is_cargo_packed = None
            cc_positions = None
            cc_rotation_mats = None
            cc_dims = None
        else:    
            is_cargo_packed = solution.cargo_container_maps >=0
            cc_positions = solution.positions[is_cargo_packed,:]
            cc_rotation_mats = solution.rotation_mats[is_cargo_packed,:,:]
            cc_dims = solution.cargo_dims[is_cargo_packed,:]
        return visualize_box(container_dim, cc_positions, cc_dims, cc_rotation_mats)

    @property
    def volume_packed_ratio(self):
        total_volume = np.sum(self.cargo_volumes)
        total_packed_volume = 0
        for solution in self.solution_list:
            is_packed = solution.cargo_container_maps>=0
            packed_volume = np.sum(solution.cargo_volumes[is_packed])
            total_packed_volume += packed_volume
        return total_packed_volume/total_volume
    
    
    @property
    def weight_packed_ratio(self):
        total_weight = np.sum(self.cargo_weights)
        total_packed_weight = 0
        for solution in self.solution_list:
            is_packed = solution.cargo_container_maps>=0
            packed_weight = np.sum(solution.cargo_weights[is_packed])
            total_packed_weight += packed_weight
        return total_packed_weight/total_weight
    
    @property
    def container_utilities(self):
        cargo_volumes = self.cargo_volumes
        container_filled_volumes = self.x.dot(cargo_volumes)
        return container_filled_volumes/self.max_container_volume

    @property
    def is_feasible(self):
        return self.is_all_cargo_packed and self.is_all_cog_feasible

    @property
    def score(self):
        return compute_obj(self.x,
                           self.cargo_volumes,
                           self.cargo_prices,
                           self.container_cost,
                           self.max_container_volume, 
                           self.omega)

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
            if not solution.is_all_cargo_packed:
                return False
        return True
    
    @property
    def packing_feasibility_ratio(self):
        num_used_container = np.sum(np.any(self.x, axis=1))
        num_container_feasible = 0.
        for solution in self.solution_list:
            if solution.is_all_cargo_packed  and len(solution.cargo_types)>0:
                num_container_feasible += 1
        return num_container_feasible/num_used_container
    
    @property
    def is_all_cog_feasible(self):
        for solution in self.solution_list:
            if not solution.is_cog_feasible:
                return False
        return True
    
    @property
    def cog_feasibility_ratio(self):
        num_used_container = np.sum(np.any(self.x, axis=1))
        num_container_feasible = 0.
        for solution in self.solution_list:
            if solution.is_cog_feasible and len(solution.cargo_types)>0:
                num_container_feasible += 1
        return num_container_feasible/num_used_container
    
def create_copy(eval_result:EvaluationResult)->EvaluationResult:
    new_x = eval_result.x.copy()
    new_solution_list = deepcopy(eval_result.solution_list)
    new_eval_result = EvaluationResult(eval_result.df_cargos,
                                       eval_result.df_containters,
                                       new_x,
                                       new_solution_list,
                                       eval_result.omega,
                                       eval_result.cargo_loads)
    return new_eval_result

def is_better(eval_result_a:EvaluationResult, eval_result_b: EvaluationResult):
    if eval_result_a.packing_feasibility_ratio > eval_result_b.packing_feasibility_ratio:
        return True
    if eval_result_a.packing_feasibility_ratio < eval_result_b.packing_feasibility_ratio:
        return False
    
    if eval_result_a.cog_feasibility_ratio > eval_result_b.packing_feasibility_ratio:
        return True
    if eval_result_a.cog_feasibility_ratio < eval_result_b.packing_feasibility_ratio:
        return False
    
    if eval_result_a.score > eval_result_b.score:
        return True
    return False
    
    