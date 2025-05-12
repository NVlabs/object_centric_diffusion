BASE_DIR=$(pwd)
TASK_DATASET_PATH=/tmp/pour_water/r3d/
YOLO_WORLD_DIR=/tmp/YOLO-World/

for task_name in \
    pour_water \
    # task2
    # task3
do
    # extract RGBD frame from .r3d file (recorded video by Record3D app)
    conda run -n spot --no-capture-output python env_real/data/prepare_rgbd.py $task_name ${TASK_DATASET_PATH}

    # get first frame segmentation from YOLO-World in "yolo_world" conda envoriment
    # *must use absolute path for some reasons
    cd ${YOLO_WORLD_DIR}
    conda run -n yolo_world --no-capture-output PYTHONPATH=${YOLO_WORLD_DIR} python ${BASE_DIR}/env_real/data/prepare_mask.py $task_name ${TASK_DATASET_PATH}

    # get grasp/trarget object pose throughout the sequence 
    cd ${BASE_DIR}
    conda run -n spot --no-capture-output python env_real/data/prepare_pose.py $task_name ${TASK_DATASET_PATH}
    # convert the data to zarr for training
    cd ${BASE_DIR}
    conda run -n spot --no-capture-output python env_real/data/collect_zarr_real.py $task_name ${TASK_DATASET_PATH}
done