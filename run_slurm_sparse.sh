#!/bin/bash
srun --gpus=1\
 --mem-per-gpu=32G\
 --cpus-per-gpu=16\
 --qos=eaton-high\
 --time=72:10:00\
 --partition=eaton-compute \
bash -c "source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_sparse.yml --dataset_path /scratch/kvedder/habitat_sampling --pipeline ObjectDetection"