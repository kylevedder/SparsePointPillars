#!/bin/bash
srun --gpus=1\
 --mem-per-gpu=24G\
 --cpus-per-gpu=16\
 --qos=eaton-high\
 --time=18:10:00\
 --partition=eaton-compute \
 --exclude=node-1080ti-3\
 --container-mounts=/scratch:/scratch,/home/kvedder/code/Open3D-ML:/project\
 --container-image=docker-registry.grasp.cluster#open3dml \
bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_dense_01_wd_4_mean_box.yml --dataset_path /scratch/kvedder/habitat_sampling_large --pipeline ObjectDetection"
