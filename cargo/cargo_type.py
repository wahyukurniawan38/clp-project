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
 z/h (top, bottom)|                  |  +
                  |                  | /
                  |                  |/ y/w (right, left)
                  +------------------+
                          x/l (front-back)
"""
class CargoType:
	def __init__(self,
                 id: str,
                 dim: np.ndarray,
                 is_dim_allow_vertical: np.ndarray,
                 weight: float,
                 cost_per_cbm: float,
                 num_cargo: int,
                 dim_r: np.ndarray = None,
                 volume: float = None,
                 rotation_mat: np.ndarray = None):
		self.id = id
		self.cost_per_cbm = cost_per_cbm
		self.dim = dim
		self.num_cargo = num_cargo
		self.dim_r = np.tile(dim, (3,1)) if dim_r is None else dim_r
		self.volume = np.prod(dim) if volume is None else volume
		self.cost = cost_per_cbm*self.volume
		self.is_dim_allow_vertical = is_dim_allow_vertical
		self.rotation_mat = np.eye(3,3, dtype=float) if rotation_mat is None else rotation_mat
		self.weight = weight

	def __str__(self):
		return str(self.__dict__)