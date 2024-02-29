import numpy as np

from solver.problem import Problem

def init_cog(cog:np.ndarray=None):
    if cog is not None:
        return cog.copy()
    return np.zeros([2,])

def init_block_position(block_position:np.ndarray=None):
    if block_position is not None:
        return block_position.copy()
    return np.full([3,],-1)

def init_num_cargo_used(problem:Problem, num_cargo_used:np.ndarray=None):
    if num_cargo_used is not None:
        return num_cargo_used.copy()
    num_cargo_used = np.zeros([len(problem.cargo_type_list),], dtype=int)
    return num_cargo_used

def init_dim(dim:np.ndarray=None):
    if dim is not None:
        return dim.copy()
    return np.zeros([3,])

def init_packing_area(packing_area:np.ndarray=None):
    if packing_area is not None:
        return packing_area.copy()
    return None