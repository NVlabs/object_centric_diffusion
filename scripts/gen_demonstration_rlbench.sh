# peract setting
for task_name in \
    meat_off_grill \
    place_wine_at_rack_location \
    insert_onto_square_peg \
    put_groceries_in_cupboard \
    place_shape_in_shape_sorter \
    reach_and_drag \
    put_money_in_safe \
    turn_tap \
    light_bulb_in \
    close_jar
do
    # training set
    python env_rlbench_peract/data/collect_zarr_rlbench_peract.py \
        --peract_demo_dir=/tmp/peract/raw/ \
        --save_path=/tmp/rlbench_zarr/ \
        --tasks=$task_name --variations=-1 --processes=1 --split=train --episodes_per_task=100

    # testing set
    python env_rlbench_peract/data/collect_zarr_rlbench_peract.py \
        --peract_demo_dir=/tmp/peract/raw/ \
        --save_path=/tmp/rlbench_zarr/ \
        --tasks=$task_name --variations=-1 --processes=1 --split=test --episodes_per_task=25
done


# peract setting (multi-object task)
for task_name in \
    stack_cups \
    stack_blocks \
    place_cups
do
    # training set
    python env_rlbench_peract/data/collect_zarr_rlbench_peract_multi_stage.py \
        --peract_demo_dir=/tmp/peract/raw/ \
        --save_path=/tmp/rlbench_zarr/ \
        --tasks=$task_name --variations=-1 --processes=1 --split=train --episodes_per_task=100

    # testing set
    python env_rlbench_peract/data/collect_zarr_rlbench_peract_multi_stage.py \
        --peract_demo_dir=/tmp/peract/raw/ \
        --save_path=/tmp/rlbench_zarr/ \
        --tasks=$task_name --variations=-1 --processes=1 --split=test --episodes_per_task=25
done