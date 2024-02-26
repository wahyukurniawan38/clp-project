import random

import json

def read_available_item_types():
    available_item_types = []
    with open("raw_data/raw.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            values = line.split()
            d_item_type = {
                "length":float(values[1]),
                "width":float(values[2]),
                "height":float(values[3]),
                "volume":float(values[4]),
                "weight":float(values[6]),
                "cost":float(values[8]),
            }
            available_item_types += [d_item_type]
    return available_item_types

if __name__ == "__main__":
    num_40_feet_container = 0
    num_20_feet_container = 2
    num_item_types = 40
    target_utility = 0.9
    instance_name = "n40_"+str(num_40_feet_container)+"_n20_"+str(num_20_feet_container)+"_nc_"+str(num_item_types)+"_u_"+str(target_utility)+".json"
    target_volume = target_utility*(num_40_feet_container*1220*240*260 + num_20_feet_container*600*240*260)
    target_weight = target_utility*(num_40_feet_container*27000+ num_20_feet_container*13000)
    container_types_list = [
        {
            "length":1220,
            "width":240,
            "height":260,
            "cost":2000,
            "max_weight":27000,
            "num":num_40_feet_container
        },
        {
            "length":600,
            "width":240,
            "height":260,
            "cost":1200,
            "max_weight":13000,
            "num":num_20_feet_container
        }
    ]
    
    cargo_types_list = []
    available_item_types = read_available_item_types()
    for i in range(num_item_types):
        chosen_idx = random.randint(0, len(available_item_types)-1)
        chosen_type = available_item_types[chosen_idx]
        chosen_type["num"]=0
        cargo_types_list += [chosen_type]
        available_item_types = available_item_types[:chosen_idx] + available_item_types[chosen_idx+1:]

    filled_volume, filled_weight = 0,0
    possible_item_type_idx = list(range(num_item_types))
    while len(possible_item_type_idx)>0:
        # remove type that cannot be incremented anymore
        for idx in possible_item_type_idx:
            if cargo_types_list[idx]["weight"] + filled_weight > target_weight or cargo_types_list[idx]["volume"] + filled_volume > target_volume:
                possible_item_type_idx.remove(idx)
        if len(possible_item_type_idx) == 0:
            break
        random_idx = random.choice(possible_item_type_idx)
        cargo_types_list[random_idx]["num"] += 1
        filled_weight += cargo_types_list[random_idx]["weight"]
        filled_volume += cargo_types_list[random_idx]["volume"]


    d = {}
    d["container_types"] = container_types_list
    d["cargo_types"] = cargo_types_list
    with open(instance_name, "w") as json_file:
        json.dump(d, json_file, indent=5)