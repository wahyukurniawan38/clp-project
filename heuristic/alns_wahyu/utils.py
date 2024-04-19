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

