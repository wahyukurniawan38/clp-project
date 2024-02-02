from typing import List

import numpy as np

rotation_list = [
        [[1,0,0],
        [0,1,0],
        [0,0,1]],
        [[1,0,0],
        [0,0,1],
        [0,1,0]],
        [[0,1,0],
        [1,0,0],
        [0,0,1]],
        [[0,1,0],
        [0,0,1],
        [1,0,0]],
        [[0,0,1],
        [1,0,0],
        [0,1,0]],
        [[0,0,1],
        [0,1,0],
        [1,0,0]]]
rotation_list = [np.asanyarray(rotation_list[i], dtype=float) for i in range(6)]

"""
       +------------------+ 
      /                  /|
     /                  / |
    +------------------+  |
    |                  |  |
 z/h|                  |  +
    |                  | /
    |                  |/ y/w
    +------------------+
            x/l 
"""
class CargoType:
    def __init__(self,
                 id: str,
                 dim: np.ndarray,
                 is_dim_allow_vertical: np.ndarray,
                 weight: float,
                 cost: float,
                 num_cargo: int,
                 dim_r: np.ndarray = None,
                 volume: float = None,
                 rotation_mat: np.ndarray = None):
        self.id = id
        self.cost = cost
        self.dim = dim
        self.num_cargo = num_cargo
        self.dim_r = np.tile(dim, (3,1)) if dim_r is None else dim_r
        self.volume = np.prod(dim) if volume is None else volume
        self.is_dim_allow_vertical = is_dim_allow_vertical
        self.rotation_mat = np.eye(3,3, dtype=float) if rotation_mat is None else rotation_mat
        self.weight = weight

    # @property
    # def size(self):
    #     current_dim = self.dim_r * self.rotation_mat
    #     current_dim = current_dim.sum(axis=1)
    #     return current_dim
    
    
    
    
# def generate_possible_rotations(box: Box) -> List[Box]:
#     idav = box.is_dim_allow_vertical
#     n_box_list: List[Box] = []
#     for r in rotation_list:
#         is_rotation_allowed = (r[-1,:]*idav).sum() > 0
#         if not is_rotation_allowed:
#             continue
#         n_box = Box(box.id,
#                     box.dim,
#                     box.is_dim_allow_vertical,
#                     box.weight,
#                     box.dim_r,
#                     box.volume,
#                     r)
#         n_box_list += [n_box]
#     return n_box_list