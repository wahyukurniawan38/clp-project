from typing import List

import numpy as np
import pandas as pd

from cargo.cargo_type import CargoType
from container.container_type import ContainerType
from solver.problem import Problem
from solver.solution import SolutionBase


def read_excel_data(data_path: str, num_items:int=None):
    df_items = pd.read_excel(data_path,sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path,sheet_name='Container', header=0)
    if num_items is not None:
        df_items = df_items[:num_items,:]
    return df_items, df_containers


"""
    df_cargos: dataframe with cargo data from excel
    cargo_mask: binary mask to determine which cargo to convert
        into cargo_type size=(N,)
"""
def create_cargo_type_list(df_cargos:pd.DataFrame, cargo_mask:np.ndarray):
    cargo_type_list = []
    idav = np.asanyarray([1,1,1], dtype=int)
    for i, row in df_cargos.iterrows():
        if cargo_mask[i]:  # Check if item is selected in any container
            dim = np.array([row['length'], row['width'], row['high']])
            cargo_type = CargoType(
                id=i,
                dim=dim,
                is_dim_allow_vertical=idav,
                weight=row['weight'],
                cost_per_cbm=row['price'],  # asumsikan price adalah cost
                num_cargo=int(row['qty']),  # asumsikan satu item per tipe
                volume=dim.prod()  # volume dihitung dari dimensi
            )
            cargo_type_list.append(cargo_type)
    return cargo_type_list

def create_container_type_list(df_containers):
    container_type_list = []
    for i, row in df_containers.iterrows():
        dim = np.array([row['length'], row['width'], row['high']])
        container_type = ContainerType(
            id=row['Container'],
            dim=dim,
            max_weight=row['weight'],
            cost=row['e'],
            cog_tolerance=[row['teta'], row['teta']],
            psi_x=row['teta'],
            psi_y=row['teta'],# asumsikan 'e' sebagai cost
            num_available=row['num']  # jumlah kontainer
        )
        container_type_list.append(container_type)
    return container_type_list


def remove_unpacked_cargo_from_solution(solution: SolutionBase)->SolutionBase:
    unpacked_cargo_idx = np.where(solution.cargo_container_maps<0)[0]
    packed_cargo_mask = solution.cargo_container_maps>=0
    unpacked_cargo_ids = solution.cargo_type_ids[unpacked_cargo_idx]
    container_type_list = solution.problem.container_type_list
    cargo_type_list = [ctype for ctype in solution.problem.cargo_type_list if ctype.id not in unpacked_cargo_ids]
    new_problem = Problem(cargo_type_list, container_type_list)

    solution.problem = new_problem
    solution.cargo_dims =  new_problem.cargo_dims
    solution.cargo_types = new_problem.cargo_types
    solution.cargo_weights = new_problem.cargo_weights
    solution.cargo_costs = new_problem.cargo_costs
    solution.cargo_volumes = new_problem.cargo_volumes
    solution.cargo_type_ids = np.asanyarray([ct.id for ct in new_problem.cargo_type_list]).astype(int)

    solution.positions = solution.positions[packed_cargo_mask,:]
    solution.cargo_container_maps = solution.cargo_container_maps[packed_cargo_mask]
    solution.rotation_mats = solution.rotation_mats[packed_cargo_mask,:,:]
    return solution