# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from rlbench.const import colors
from rlbench.tasks.put_groceries_in_cupboard import GROCERY_NAMES
from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES

dustpan_sizes = ['tall', 'short']

task_object_dict = {
    "meat_off_grill": {
        "grasp_object_name": {
            0: "chicken",
            1: "steak"
        },
        "target_object_name": "grill",
    },
    "turn_tap": {
        "grasp_object_name": {
            0: "tap_left",
            1: "tap_right"
        },
        "target_object_name": "tap_main",
    },
    "close_jar": {
        "grasp_object_name": "jar_lid0", #{i: f"jar_lid{i % 2}" for i in range(len(colors))},
        "target_object_name": {i: f"jar{i % 2}" for i in range(len(colors))},
    },
    "reach_and_drag": {
        "grasp_object_name": "stick",
        "target_object_name": "target0",
    },
    "stack_blocks": { # multiple stage
        "grasp_object_name": {i: f"stack_blocks_target{i}" for i in range(4)},
        "target_object_name": {
            0: "stack_blocks_target_plane",
            1: f"stack_blocks_target{0}",
            2: f"stack_blocks_target{1}",
            3: f"stack_blocks_target{2}",
        },
    },
    "light_bulb_in": {
        "grasp_object_name": {i: f"light_bulb{i % 2}" for i in range(len(colors))},
        "target_object_name": "lamp_base",
    },
    "put_money_in_safe": {
        "grasp_object_name": "dollar_stack",
        "target_object_name": "safe_body",
    },
    "place_wine_at_rack_location": {
        "grasp_object_name": "wine_bottle",
        "target_object_name": "rack_top",
    },
    "put_groceries_in_cupboard":{
        "grasp_object_name": {i: GROCERY_NAMES[i].replace(' ', '_') for i in range(len(GROCERY_NAMES))},
        "target_object_name": "cupboard",
    },
    "place_shape_in_shape_sorter":{
        "grasp_object_name": {i: SHAPE_NAMES[i].replace(' ', '_') for i in range(len(SHAPE_NAMES))},
        "target_object_name": "shape_sorter",
    },
    "insert_onto_square_peg":{
        "grasp_object_name": "square_ring",
        "target_object_name": "__NONE__",   # handled on run time
    },
    "stack_cups": { # multiple stage
        "grasp_object_name": {
            0: "cup1",
            1: "cup3",
        },
        "target_object_name": {
            0: "cup2",
            1: "cup1",
        },
    },
    "place_cups": { # multiple stage
        "grasp_object_name": {i: f"mug{i}" for i in range(3)},
        "target_object_name": "place_cups_holder_base",
    },
}