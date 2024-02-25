import json
from typing import List

import numpy as np

from cargo.cargo_type import CargoType
from container.container_type import ContainerType

"""
    Many container types
        many containers each
    Many cargo types
        many cargo each
"""
class Problem:
   
    def __init__(self,
                 cargo_type_list: List[CargoType],
                 container_type_list: List[ContainerType]):
        self.cargo_type_list = cargo_type_list
        self.container_type_list = container_type_list

        # we can init numpy arrays here
        # about the cargos and the containers
        # shared by all solution
        self.cargo_dims =    np.asanyarray([cargo_type.dim for cargo_type in cargo_type_list for _ in range(cargo_type.num_cargo)])
        self.cargo_types =   np.asanyarray([i for i, cargo_type in enumerate(cargo_type_list) for _ in range(cargo_type.num_cargo)])
        self.cargo_weights = np.asanyarray([cargo_type.weight for cargo_type in cargo_type_list for _ in range(cargo_type.num_cargo)])
        self.cargo_costs =   np.asanyarray([cargo_type.cost for cargo_type in cargo_type_list for _ in range(cargo_type.num_cargo)])
        self.cargo_volumes = np.asanyarray([cargo_type.volume for cargo_type in cargo_type_list for _ in range(cargo_type.num_cargo)])

        # this one per types
        # useful for block building
        self.cargo_type_dims = np.asanyarray([cargo_type.dim for cargo_type in cargo_type_list])
        self.cargo_type_weights = np.asanyarray([cargo_type.weight for cargo_type in cargo_type_list])
        self.cargo_type_volumes = np.asanyarray([cargo_type.volume for cargo_type in cargo_type_list ])
        self.cargo_type_costs = np.asanyarray([cargo_type.cost for cargo_type in cargo_type_list])


def read_from_file(instance_path):
    with open(instance_path, "r") as instance_file:
        d = json.load(instance_file)
        container_type_list = []
        d_containter_types = d["container_types"]
        for i, d_container_type in enumerate(d_containter_types):
            l = d_container_type["length"]
            w = d_container_type["width"]
            h = d_container_type["height"]
            dim = np.asanyarray([l,w,h], dtype=float)
            container_type = ContainerType(str(i),
                                            dim,
                                            d_container_type["max_weight"],
                                            d_container_type["cost"],
                                            int(d_container_type["num"]))
            container_type_list += [container_type]
        d_cargo_types = d["cargo_types"]
        cargo_type_list = []
        for i, d_cargo_type in enumerate(d_cargo_types):
            l = d_cargo_type["length"]
            w = d_cargo_type["width"]
            h = d_cargo_type["height"]
            dim = np.asanyarray([l,w,h], dtype=float)
            idav = np.asanyarray([1,1,1], dtype=int)
            volume = None if "volume" not in d_cargo_type else d_cargo_type["volume"]
            cargo_type = CargoType(str(i),
                                   dim,
                                   idav,
                                   float(d_cargo_type["weight"]),
                                   float(d_cargo_type["cost"]),
                                   int(d_cargo_type["num"]),
                                   volume=volume)
            cargo_type_list += [cargo_type]
        return Problem(cargo_type_list, container_type_list)