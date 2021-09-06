#!/bin/bash
#SBATCH --qos=viz
#SBATCH --partition=viz
#SBATCH --cores=1
#SBATCH --mem=2G

PORT_MAP=/tmp/tensorboard_port_map
TB_PORT=$(cat $PORT_MAP | grep "$SLURM_JOBID," | cut -d',' -f2)
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=$1

echo "Go to http://$IP_ADDRESS:$TB_PORT"

tensorboard --bind_all --logdir $TB_FOLDER --port $TB_PORT

