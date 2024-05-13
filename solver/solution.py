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
        self.cargo_type_ids = np.asanyarray([ct.id for ct in problem.cargo_type_list]).astype(int)

        # init or deepcopy from kwargs
        self.positions = init_positions(problem, kwargs.get("positions"))
        self.cargo_container_maps = init_cargo_container_maps(problem, kwargs.get("cargo_container_maps"))
        self.rotation_mats = init_rotation_mats(problem, kwargs.get("rotation_mats"))
        
        self.nums_container_used = init_nums_container_used(problem, kwargs.get("nums_container_used"))
        self.container_dims = init_container_dims(kwargs.get("container_dims"))
        self.container_max_volumes = init_container_max_volumes(kwargs.get("container_max_volumes"))
        self.container_filled_volumes = init_container_filled_volumes(kwargs.get("container_filled_volumes"))      
        self.container_max_weights = init_container_max_weights(kwargs.get("container_max_weights"))
        self.container_filled_weights = init_container_filled_weights(kwargs.get("container_filled_weights"))
        self.container_costs = init_container_costs(kwargs.get("container_costs"))      
        self.container_types = init_container_types(kwargs.get("container_types"))
        self.container_cogs = init_container_cogs(kwargs.get("container_cogs"))
        self.container_cog_tolerances = init_container_cog_tolerances(kwargs.get("container_cog_tolerances"))

    @property
    def is_empty(self)->np.ndarray:
        return len(self.problem.cargo_type_list)==0

    @property
    def real_cargo_dims(self)->np.ndarray:
        return (self.cargo_dims[:,np.newaxis,:]*self.rotation_mats).sum(axis=-1)

    @property
    def is_all_cargo_packed(self)->bool:
        if self.is_empty:
            return True
        return np.all(self.cargo_container_maps>=0)

    @property
    def is_cog_feasible(self)->bool:
        if self.is_empty:
            return True
        min_cogs = self.container_dims[:,:2]/2 - self.container_cog_tolerances[np.newaxis,:]
        max_cogs = self.container_dims[:,:2]/2 + self.container_cog_tolerances[np.newaxis,:]
        return np.all(np.logical_and(self.container_cogs>=min_cogs, self.container_cogs<=max_cogs))

    def __str__(self) -> str:
        is_cargo_packed = self.cargo_container_maps>=0
        is_container_used = self.container_filled_volumes>0
        total_volume_packed = np.sum(self.cargo_volumes[is_cargo_packed])
        total_volume = np.sum(self.cargo_volumes)
        
        if total_volume_packed > 0:
            print("Total volume packed: ", total_volume_packed, "("+str(total_volume_packed/total_volume)+")")
        else:
            print("Total volume packed: ", total_volume_packed)
        revenue = np.sum(self.cargo_costs[is_cargo_packed])
        expense = np.sum(self.container_costs[is_container_used])
        total_profit = revenue-expense
        print("Profit: ", total_profit)
        print("Revenue: ", revenue)
        print("Expense: ", expense)
        for ct_type in range(len(self.nums_container_used)):
            is_used = self.container_filled_volumes>0
            is_this_type_used = np.logical_and(self.container_types == ct_type, is_used)
            num_this_type_used = np.count_nonzero(is_this_type_used)
            print("Number of container type ",ct_type," used:", num_this_type_used)
        for ct_idx in range(len(self.container_filled_volumes)):
            utilization = self.container_filled_volumes[ct_idx]/self.container_max_volumes[ct_idx]
            print("Container index ", ct_idx, " utilization: ", utilization)
            print("Center of gravity:", self.container_cogs[ct_idx,:])
            is_in_container = self.cargo_container_maps == ct_idx
            if not np.any(is_in_container):
                continue
            idx_in_container = np.arange(len(self.cargo_weights))[is_in_container]
            print("Items' position in container index ", ct_idx)
            print("Item x y z")
            for idx in idx_in_container:
                print(idx,self.positions[idx,0],self.positions[idx,1],self.positions[idx,2])
            
            print("Item's rotation in container index ",ct_idx)
            print("Item","lx","ly","lz","wx","wy","wz","hx","hy","hz")
            for idx in idx_in_container:
                rotation = self.rotation_mats[idx]
                lx,ly,lz = rotation[0,0],rotation[0,1],rotation[0,2]
                wx,wy,wz = rotation[1,0],rotation[1,1],rotation[1,2]
                hx,hy,hz = rotation[2,0],rotation[2,1],rotation[2,2]
                print(self.cargo_type_ids[idx],lx,ly,lz,wx,wy,wz,hx,hy,hz)
        return ""
            
            