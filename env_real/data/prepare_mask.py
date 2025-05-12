# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
sys.path.insert(0, "/tmp/SPOT/")

# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import argparse
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       use_amp=False,
                       show=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(image_path)
    anno_image = image.copy()
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)

    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image_path)] = anno_image
        annotations_dict[osp.basename(image_path)] = detections

        ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=texts, images=images_dict,
            annotations=annotations_dict).as_yolo(
                annotations_directory_path=ANNOTATIONS_DIRECTORY,
                min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
                max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
                approximation_percentage=APPROXIMATION_PERCENTAGE)

    if show:
        cv2.imshow('Image', image)  # Provide window name
        k = cv2.waitKey(0)
        if k == 27:
            # wait for ESC key to exit
            cv2.destroyAllWindows()
    return pred_instances['bboxes'], image


if __name__ == '__main__':
    from pathlib import Path
    from zipfile import ZipFile
    import glob
    import shutil
    import numpy as np
    from env_real.utils.realworld_objects import real_task_object_dict

    # setup yolo world
    yolo_world_dir = "/tmp/YOLO-World/"
    config_path = f"{yolo_world_dir}/configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py"
    weight_path = f"{yolo_world_dir}/weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
    topk = 1
    threshold = 0.005
    amp = False
    show = False
    annotation = False

    # load config
    cfg = Config.fromfile(config_path)
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(config_path))[0])
    # init model
    cfg.load_from = weight_path
    model = init_detector(cfg, checkpoint=weight_path, device='cuda:0')

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    # load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task_name', type=str)
    parser.add_argument('dataset_path', default="/tmp/record3d/[task_name]/r3d/", type=str)
    args = parser.parse_args()

    task_name = args.task_name
    print(task_name)
    dataset_path = args.dataset_path
    print(dataset_path)
    r3d_path_list = glob.glob(os.path.join(dataset_path, '*.r3d'))
    r3d_path_list.sort()

    for obj_type in ['grasp', 'target']:
        
        # setup text keyword
        prompts = real_task_object_dict[task_name][f"{obj_type}_object_prompt"]
        texts = [[t] for t in prompts] + [[' ']]
        
        for r3d_file in r3d_path_list:
            print(r3d_file)
            
            extract_path = os.path.join(dataset_path, Path(r3d_file).stem)
            rgb_dir = os.path.join(extract_path, 'rgb')

            # only the first frame
            frame_idx = 0
            rgb_path = os.path.join(rgb_dir, f'{frame_idx}.jpg')
            fname = Path(rgb_path).stem

            # run inference
            vis_dir = os.path.join(extract_path, f'vis')
            os.makedirs(vis_dir, exist_ok=True)
            model.reparameterize(texts)
            bboxes, vis_image = inference_detector(model,
                            rgb_path,
                            texts,
                            test_pipeline,
                            topk,
                            threshold,
                            use_amp=amp,
                            show=show,
                            annotation=annotation)
            cv2.imwrite((os.path.join(vis_dir, f'bbox_{obj_type}.png')), vis_image)
            
            # get mask
            bbox = bboxes[0]    # !! assume only one box
            rgb = cv2.imread(rgb_path)
            mask = np.zeros(rgb.shape[:-1])
            x1, y1, x2, y2 = bbox.astype(np.int)
            mask[y1:y2, x1:x2] = 1.
            # print(np.unique(mask, return_counts=True))

            # save the mask
            mask_dir = os.path.join(extract_path, f'mask_{obj_type}')
            os.makedirs(mask_dir, exist_ok=True)

            pallet = np.array([
                [0, 0, 0], 
                [255, 255, 255]
            ]).astype(np.uint8)

            from PIL import Image
            mask_image = Image.fromarray(mask.astype(np.uint8)).convert('P')
            mask_image.putpalette(pallet)
            mask_image.save(os.path.join(mask_dir, fname+'.png'))

            # read the mask
            # mask_image = Image.open(os.path.join(mask_dir, fname+'.png')).convert('P')
            # mask = np.array(mask_image)
            # print(np.unique(mask, return_counts=True))
            # mask_image.close()