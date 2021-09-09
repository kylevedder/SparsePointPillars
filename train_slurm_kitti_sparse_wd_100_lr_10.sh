#!/bin/bash
srun --gpus=1\
 --mem-per-gpu=20G\
 --cpus-per-gpu=12\
 --qos=eaton-high\
 --time=48:10:00\
 --partition=eaton-compute \
 --container-mounts=/scratch:/scratch,/Datasets:/Datasets,/home/kvedder/code/Open3D-ML:/project\
 --container-image=docker-registry.grasp.cluster#open3dml \
bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/sparse_pointpillars_kitti_car_only_wd_100_lr_10.yml --dataset_path /Datasets/kitti_object_detect/ --pipeline ObjectDetection"
