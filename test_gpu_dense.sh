#!/bin/bash
source set_open3d_ml_root.sh; OMP_NUM_THREADS=1 python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling.yml --dataset_path /data/habitat_sampling_dataset/ --pipeline ObjectDetection --split test