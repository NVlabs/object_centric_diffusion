# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from pyrep.const import RenderMode
from pyrep.objects import VisionSensor

def get_camera(name):
    if name == "custom_cam_front_light_bulb_in":
        cam = VisionSensor.create([1024, 576])
        cam.set_explicit_handling(True)
        # cam.set_position([1.0, 0., 1.3])
        cam.set_position([0.9, 0., 1.3])
        cam.set_orientation([-np.pi, -0.4*np.pi, 0.5*np.pi])
        cam.set_render_mode(RenderMode.OPENGL)
    else:
        raise NotImplementedError
    return cam