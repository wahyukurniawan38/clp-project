import numpy as np

class ContainerType:
    def __init__(self,
                 id: str,
                 dim:np.ndarray,
                 max_weight:np.ndarray,
                 cost:float,
                 cog_tolerance:np.ndarray,
                 psi_x:float,
                 psi_y:float,
                 num_available:int=999999,):
        self.id:str = id
        self.dim:np.ndarray = dim
        self.max_volume:float = np.prod(dim)
        self.max_weight:float = max_weight
        self.num_available:int = num_available
        self.cog_tolerance:np.ndarray = cog_tolerance
        self.psi_x:float = psi_x
        self.psi_y:float = psi_y# [psi_x, psi_y]
        self.cost:float = cost