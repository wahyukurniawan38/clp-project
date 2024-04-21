import numpy as np

class RandomRepair:
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
        unpacked_cargo_idx = np.where(np.logical_not(np.any(x, axis=0)))[0]
        container_filled_weights = x.dot(cargo_weights)
        container_filled_volumes = x.dot(cargo_volumes)
        for c_idx in unpacked_cargo_idx:
            container_vol_after_insert = container_filled_volumes + cargo_volumes[c_idx]
            container_weight_after_insert = container_filled_weights + cargo_weights[c_idx]
            is_ct_feasible_to_insert = np.logical_and(container_vol_after_insert<=container_max_volume, container_weight_after_insert<=container_max_weight)
            if not np.any(is_ct_feasible_to_insert):
                continue
            feasible_ct_idx = np.where(is_ct_feasible_to_insert)[0]
            chosen_ct_idx = np.random.choice(feasible_ct_idx)
            x[chosen_ct_idx,c_idx] = 1
            container_filled_volumes[chosen_ct_idx] += cargo_volumes[c_idx]
            container_filled_weights[chosen_ct_idx] += cargo_weights[c_idx]
        return x
    
    def __str__(self) -> str:
        return "Random Repair Operator"
    

class GreedyRepair:
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
        is_cargo_not_packed = np.logical_not(np.any(x, axis=0))
        unpacked_cargo_idx = np.where(is_cargo_not_packed)[0]
        unpacked_cargo_sorted_idx = unpacked_cargo_idx[np.argsort(-cargo_ratios[unpacked_cargo_idx])]
        container_filled_weights = x.dot(cargo_weights)
        container_filled_volumes = x.dot(cargo_volumes)
        for c_idx in unpacked_cargo_sorted_idx:
            container_vol_after_insert = container_filled_volumes + cargo_volumes[c_idx]
            container_weight_after_insert = container_filled_weights + cargo_weights[c_idx]
            is_ct_feasible_to_insert = np.logical_and(container_vol_after_insert<=container_max_volume, container_weight_after_insert<=container_max_weight)
            if not np.any(is_ct_feasible_to_insert):
                continue
            feasible_ct_idx = np.where(is_ct_feasible_to_insert)[0]
            feasible_ct_vol = container_filled_volumes[feasible_ct_idx]
            feasible_ct_sorted_idx_by_vol = np.argsort(feasible_ct_vol)
            feasible_ct_idx = feasible_ct_idx[feasible_ct_sorted_idx_by_vol]
            is_container_filled = container_filled_volumes>0
            is_container_filled = is_container_filled[feasible_ct_sorted_idx_by_vol]
            if np.any(is_container_filled):
                feasible_ct_idx = feasible_ct_idx[is_container_filled]
            chosen_ct_idx = feasible_ct_idx[0]
            x[chosen_ct_idx,c_idx] = 1
            container_filled_volumes[chosen_ct_idx] += cargo_volumes[c_idx]
            container_filled_weights[chosen_ct_idx] += cargo_weights[c_idx]
        return x 
    
    def __str__(self) -> str:
        return "Greedy Insertion Repair Operator"