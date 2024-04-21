import numpy as np

class RandomRemoval:
    def __init__(self) -> None:
        return

    def __call__(self, 
                 x: np.ndarray, 
                 cargo_volumes:np.ndarray,
                 cargo_weights:np.ndarray,
                 cargo_ratios:np.ndarray,
                 cargo_loads:np.ndarray,
                 container_max_volume:float,
                 container_max_weight:float,
                 *args,
                 **kwargs) -> np.ndarray:
        nof = kwargs["nof"]
        item_indices = np.flatnonzero(x)  # Mengambil semua indeks item yang dipilih
        items_to_remove = np.random.choice(item_indices, size=nof, replace=False)  # Memilih item yang akan dihapus
        items_to_remove_2d = np.unravel_index(items_to_remove, x.shape)
        x[items_to_remove_2d] = 0
        return x
    
    def __str__(self) -> str:
        return "Random Removal Destroy Operator"
        

def remove_cargo_worst_ratio(x: np.ndarray, ratio: np.ndarray, nof:float):
    is_container_filled = np.sum(x, axis=1) > 0
    filled_container_idx = np.where(is_container_filled)[0]
    for container_idx in filled_container_idx:
        container_items = x[container_idx, :]
        item_indices = np.where(container_items == 1)[0]
        sorted_indices = item_indices[np.argsort(ratio[item_indices])]
    items_to_zero = sorted_indices[:nof]
    x[container_idx, items_to_zero] = 0
    return x

def empty_container_worst_ratio(x:np.ndarray, ratio: np.ndarray, nof2: int):
    container_total_ratios = np.dot(x, ratio)
    is_container_filled = np.any(x, axis=1)
    filled_container_idx = np.where(is_container_filled)[0]
    filled_container_ratio = container_total_ratios[filled_container_idx]
    sorted_idx_ = np.argsort(filled_container_ratio)
    filled_container_idx = filled_container_idx[sorted_idx_]
    filled_container_idx = filled_container_idx[:nof2]
    x[filled_container_idx,:] = 0
    return x

class WorstRemoval:
    def __init__(self) -> None:
        return 

    def __call__(self,
                 x:np.ndarray,
                 cargo_volumes:np.ndarray,
                 cargo_weights:np.ndarray,
                 cargo_ratios:np.ndarray,
                 cargo_loads:np.ndarray,
                 container_max_volume:float,
                 container_max_weight:float,
                 *args,
                 **kwargs)->np.ndarray:
        nof = kwargs["nof"]
        nof2 = kwargs["nof2"]
        x = remove_cargo_worst_ratio(x, cargo_ratios, nof)
        if nof2 > 0:
            x = empty_container_worst_ratio(x, cargo_ratios, nof2)
        return x
    
    def __str__(self) -> str:
        return "Worst Removal Destroy Operator"


