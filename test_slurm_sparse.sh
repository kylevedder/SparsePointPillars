#!/bin/bash
srun --gpus=1\
 --mem-per-gpu=32G\
 --cpus-per-gpu=16\
 --qos=eaton-high\
 --time=00:10:00\
 --partition=eaton-compute \
 --exclude=node-1080ti-3\
 --container-mounts=/scratch:/scratch,/home/kvedder/code/Open3D-ML:/project\
 --container-image=docker-registry.grasp.cluster#open3dml \
bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_sparse_01.yml --dataset_path /scratch/kvedder/habitat_sampling_large --pipeline ObjectDetection --split test --ckpt_path=./logs_sparse_01_wd_4_2x2/SparsePointPillars_HabitatSampling_torch/checkpoint/ckpt_00300.pth"
#bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_sparse_01.yml --dataset_path /scratch/kvedder/habitat_sampling_large --pipeline ObjectDetection --split test --ckpt_path=./logs_sparse_01_wd_4/SparsePointPillars_HabitatSampling_torch/checkpoint/ckpt_00300.pth"
#bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_sparse_01.yml --dataset_path /scratch/kvedder/habitat_sampling_large --pipeline ObjectDetection --split test --ckpt_path=./logs_sparse_01/SparsePointPillars_HabitatSampling_torch/checkpoint/ckpt_00500.pth"
