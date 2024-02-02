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

def read_from_file(instance_path):
    with open(instance_path, "r") as instance_file:
        lines = instance_file.readlines()
        num_of_container_type = int(lines[0].split()[1])
        container_type_list = []
        for i in range(1,num_of_container_type+1):
            line = lines[i].split()
            dim = np.asanyarray([float(line[1]),float(line[3]),float(line[5])], dtype=float)
            
            container_type = ContainerType(str(i),
                                   dim,
                                   float(line[7]),
                                   int(line[9]))
            container_type_list += [container_type]
        
        num_cargo_type = int(lines[num_of_container_type+1].split()[1])
        cargo_type_list = []

        for i in range(num_of_container_type+2, len(lines)):
            line = lines[i].split()
            dim = np.asanyarray([float(line[1]),float(line[3]),float(line[5])], dtype=float)
            idav = np.asanyarray([1,1,1], dtype=int)
            cargo_type = CargoType(str(i),
                                   dim,
                                   idav,
                                   float(line[7]),
                                   float(line[9]),
                                   int(line[11]))
            cargo_type_list += [cargo_type]
        return Problem(cargo_type_list, container_type_list)