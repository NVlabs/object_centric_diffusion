# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import numpy as np


def save_np_dict_to_json(np_dict, save_path):
    json_string = json.dumps({k: v.tolist() for k, v in np_dict.items()}, indent=4)
    with open(save_path, "w") as f: 
        f.write(json_string)

def load_np_dict_from_json(read_path):
    with open(read_path, "r") as f: 
        np_dict = {k: np.array(v) for k, v in json.load(f).items()}
    return np_dict


if __name__ == "__main__":
    # Test data
    d = {
    'chicken': np.random.randn(5),
    'banana': np.random.randn(5),
    'carrots': np.random.randn(5)
    }

    save_path = "test.json"
    save_np_dict_to_json(d, save_path)
    a = load_np_dict_from_json(save_path)
    
    print (a)
    print (d)