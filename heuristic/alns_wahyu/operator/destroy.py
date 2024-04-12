import numpy as np

class RandomRemoval:
    def __init__(self, nof: int) -> None:
        self.nof = nof

    def __call__(self, ccm: np.ndarray, nof: int=None) -> np.Any:
        nof = nof or self.nof
        item_indices = np.flatnonzero(ccm)  # Mengambil semua indeks item yang dipilih
        items_to_remove = np.random.choice(item_indices, size=nof, replace=False)  # Memilih item yang akan dihapus
        items_to_remove_2d = np.unravel_index(items_to_remove, ccm.shape)
        ccm[items_to_remove_2d] = 0
        return ccm
        

def remove_cargo_worst_ratio(ccm: np.ndarray, ratio: np.ndarray, nof:float):
    is_container_filled = np.sum(ccm, axis=1) > 0
    filled_container_idx = np.where(is_container_filled)[0]
    print('container has items:',filled_container_idx)
    for container_idx in filled_container_idx:
        container_items = ccm[container_idx, :]
        item_indices = np.where(container_items == 1)[0]
        sorted_indices = item_indices[np.argsort(ratio[item_indices])]
    items_to_zero = sorted_indices[:nof]
    ccm[container_idx, items_to_zero] = 0
    return ccm

def empty_container_worst_ratio(ccm:np.ndarray, ratio: np.ndarray, nof2: int):
    container_total_ratios = np.dot(ccm, ratio)
    is_container_filled = np.any(ccm, axis=1)
    filled_container_idx = np.where(is_container_filled)[0]
    filled_container_ratio = container_total_ratios[filled_container_idx]
    sorted_idx_ = np.argsort(filled_container_ratio)
    filled_container_idx = filled_container_idx[sorted_idx_]
    filled_container_idx = filled_container_idx[:nof2]
    ccm[filled_container_idx,:] = 0
    return ccm

class WorstRemoval:
    def __init__(self, nof:int, nof2:int) -> None:
        self.nof = nof
        self.nof2 = nof2


    def __call__(self, ccm:np.ndarray, ratio:np.ndarray, nof:int=None, nof2:int=None):
        nof = nof or self.nof
        nof2 = nof2 or self.nof2
        ccm = remove_cargo_worst_ratio(ccm, ratio, nof)
        if nof2 > 0:
            ccm = empty_container_worst_ratio(ccm, ratio, nof2)
        return ccm


