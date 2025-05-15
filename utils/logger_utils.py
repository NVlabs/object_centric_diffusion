# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import imageio
from PIL import Image
from utils.mask_utils import palette_ADE20K


def _save_rgb(data, path, type='video'):
    assert type in ['images', 'gif', 'video']

    if type == 'gif':
        # save gif
        pil_images = [Image.fromarray(img) for img in data]
        pil_images[0].save(path, save_all=True, append_images=pil_images[1:], duration=50, loop=0)  # duration is the number of milliseconds between frames; this is 40 frames per second
    elif type == 'video':
        # save video
        video_writer = imageio.get_writer(path, fps=40)
        for img in data:
            video_writer.append_data(img)
        video_writer.close()
    else:
        raise NotImplementedError

def _save_combined_rgb(data_list, path, type='video'):
    assert type in ['gif', 'video']
    # TODO: check if length is the same
    # TODO: check shape

    n_frame = len(data_list[0])
    for data in data_list:
        assert len(data) == n_frame

    data_combined = []
    for i in range(n_frame):
        data_combined.append(
            np.concatenate([frames[i] for frames in data_list], axis=1)
        )

    if type == 'gif':
        # save gif
        pil_images = [Image.fromarray(img) for img in data_combined]
        pil_images[0].save(path, save_all=True, append_images=pil_images[1:], duration=50, loop=0)  # duration is the number of milliseconds between frames; this is 40 frames per second
    elif type == 'video':
        # save video
        video_writer = imageio.get_writer(path, fps=40)
        for img in data_combined:
            video_writer.append_data(img)
        video_writer.close()
    else:
        raise NotImplementedError

def _save_mask(data, path, palette=None):

    print("The function _save_mask currently only saves the first image.")
    # TODO: support multiple images
    mask_image = data[0]

    if palette is None:
        palette = palette_ADE20K
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert('P')
    mask_image.putpalette(palette)
    mask_image.save(path)


class EnvLogger:
    def __init__(self):
        self.data = {}
        self.data_type = {}
    
    def clear(self):
        for name in self.data:
            self.data[name].clear()

    def add_data_type(self, name, type):
        self.data[name] = []
        self.data_type[name] = type

    def add_data(self, name, new_data):
        assert name in self.data
        self.data[name].append(new_data)
    
    def get_data(self, name):
        return self.data[name]

    def save_data(self, name, path, output_fn=None, **kwargs):
        if isinstance(name, list):
            # TODO: check if type is the same
            data_list = []
            data_type = self.data_type[name[0]]
            for k in name:
                data_list.append(self.data[k])
            output_fn = output_fn if output_fn is not None else self._get_combined_output_fn(data_type)
            output_fn(data_list, path, **kwargs)
        else:
            data = self.data[name]
            data_type = self.data_type[name]
            output_fn = output_fn if output_fn is not None else self._get_output_fn(data_type)
            output_fn(data, path, **kwargs)

    def _get_output_fn(self, type):
        assert type in ['list', 'array', 'rgb', 'depth', 'mask']
        if type == 'list':
            raise NotImplementedError
        elif type == 'array':
            raise NotImplementedError
        elif type == 'rgb':
            return _save_rgb
        elif type == 'depth':
            raise NotImplementedError
        elif type == 'mask':
            return _save_mask
        else:
            raise NotImplementedError

    def _get_combined_output_fn(self, type):
        assert type in ['list', 'array', 'rgb', 'depth', 'mask']
        if type == 'list':
            raise NotImplementedError
        elif type == 'array':
            raise NotImplementedError
        elif type == 'rgb':
            return _save_combined_rgb
        elif type == 'depth':
            raise NotImplementedError
        else:
            raise NotImplementedError