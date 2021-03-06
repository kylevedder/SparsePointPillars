#!/bin/bash
#!/bin/bash
x=1
while [ $x -le 3 ]
do
  echo "Step $x"
  source set_open3d_ml_root.sh; OMP_NUM_THREADS=`nproc` python scripts/run_pipeline.py torch -c ml3d/configs/pointpillars_habitat_sampling_dense_5000_mean_box.yml --dataset_path /data/habitat_sampling_dataset --pipeline ObjectDetection --split test --ckpt_path=./logs_dense_5000_mean_box/PointPillars_HabitatSampling_torch/checkpoint/ckpt_00200.pth --device cpu
  x=$(( $x + 1 ))
done
