import numpy as np

class ContainerType:
    def __init__(self,
                 id: str,
                 dim:np.ndarray,
                 max_weight:np.ndarray,
                 cost:float,
                 num_available:int=999999,):
        self.id:str = id
        self.dim:np.ndarray = dim
        self.max_volume:float = np.prod(dim)
        self.max_weight:float = max_weight
        self.num_available:int = num_available
        self.cost:float = cost