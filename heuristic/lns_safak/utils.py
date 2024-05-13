from solver.problem import Problem

import numba as nb
import numpy as np

from heuristic.utils import is_collide_3d, get_bottom_surface, get_top_surface, compute_collision
from solver.utils import get_possible_rotation_mats

def init_cargo_type_priority(problem:Problem, cargo_type_priority=None):
    if cargo_type_priority is not None:
        return cargo_type_priority.copy()
    cargo_type_priority = np.arange(len(problem.cargo_type_list), dtype=int)
    return cargo_type_priority


def init_cargo_type_rotation_sorted_idx(problem:Problem, cargo_type_rotation_sorted_idx=None):
    if cargo_type_rotation_sorted_idx is not None:
        return cargo_type_rotation_sorted_idx.copy()
    cargo_type_rotation_sorted_idx = np.arange(6)[np.newaxis,:]
    cargo_type_rotation_sorted_idx = np.repeat(cargo_type_rotation_sorted_idx, len(problem.cargo_type_list), axis=0)
    return cargo_type_rotation_sorted_idx


"""
    I assume sorting the rotation of each
    cargo type for each container type
    will be done very very often,
    that is why, let's pre-compute
    them, and fetch them when we need them.


    out:
        default_rotation: (n_cargo_type, n_container_type, 6)
"""
def init_default_cargo_type_rotation_sorted_idx(problem: Problem, default_cargo_type_rotation_sorted_idx=None):
    if default_cargo_type_rotation_sorted_idx is not None:
        return default_cargo_type_rotation_sorted_idx
    cargo_type_dims = [problem.cargo_type_list[i].dim[np.newaxis,:] for i in range(len(problem.cargo_type_list))]
    cargo_type_dims = np.concatenate(cargo_type_dims, axis=0)
    container_type_dims = [problem.container_type_list[i].dim[np.newaxis,:] for i in range(len(problem.container_type_list))]
    container_type_dims = np.concatenate(container_type_dims, axis=0)
    possible_rotation_mats = get_possible_rotation_mats()
    c_type_real_dims = (cargo_type_dims[:,np.newaxis,np.newaxis,:]*possible_rotation_mats[np.newaxis,:,:,:]).sum(axis=-1)
    c_type_real_wall_dims = c_type_real_dims[:,:,[1,2]]
    ct_type_wall_dims = container_type_dims[:,[1,2]]
    num_wall_span  = np.floor(ct_type_wall_dims[np.newaxis,np.newaxis,:,:]/c_type_real_wall_dims[:,:,np.newaxis,:])
    wall_span_area = np.prod(num_wall_span*c_type_real_wall_dims[:,:,np.newaxis,:],axis=-1)
    wall_span_area = np.transpose(wall_span_area, axes=(0,2,1))
    default_c_rotation_sorted_idx = np.argsort(wall_span_area,axis=-1)
    return default_c_rotation_sorted_idx

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:]), cache=True)
def filter_infeasible_addition_points(addition_points, cc_positions, cc_dims, container_dim):
    # n_ap,_ = addition_points.shape
    dummy_dim = np.full_like(addition_points, 0.00001)
    # dummy_dim = np.asanyarray([[0.00001,0.00001,0.00001]]*len(addition_points))
    # if the addition point is inside other cargo
    # cargo already placed on it
    is_collide_matrix = is_collide_3d(addition_points, dummy_dim, cc_positions, cc_dims)
    is_collide_any = np.sum(is_collide_matrix, axis=-1) > 0
    is_not_collide = np.logical_not(is_collide_any)

    # if it is on the edges of the container
    is_not_overflow = (dummy_dim + addition_points) <= container_dim[np.newaxis,:]
    is_not_overflow = np.sum(is_not_overflow, axis=-1) == 3
    
    is_feasible = is_not_collide
    is_feasible = np.logical_and(is_feasible, is_not_overflow)
    return addition_points[is_feasible]


def get_addition_points(cc_positions, cc_dims, container_dim)->np.ndarray:
    if len(cc_positions)==0:
        return np.zeros([1,3], dtype=float)
    addition_points = cc_dims[:, np.newaxis,:]*np.eye(3,3)[np.newaxis,:,:] + cc_positions[:, np.newaxis, :]
    addition_points = addition_points.reshape(len(cc_positions)*3,3)
        
    addition_points = filter_infeasible_addition_points(addition_points, cc_positions, cc_dims, container_dim)
    return addition_points

def argsort_addition_points(addition_points, mode="layer-building"):
    # if mode == "layer-building"
    criteria = (addition_points[:,1], addition_points[:,0], addition_points[:,2])
    
    if mode=="wall-building":
        criteria = (addition_points[:,1], addition_points[:,2], addition_points[:,0])
    elif mode=="column-building":
        criteria = (addition_points[:,0], addition_points[:,2], addition_points[:,1])
    sorted_idx = np.lexsort(criteria)
    return sorted_idx