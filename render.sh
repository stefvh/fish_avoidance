#!/bin/bash

PARENT_FOLDER=${1}
EXPERIMENT_DATETIME=${2}
N_TRIALS=${3}

for i in $(seq 0 1 $N_TRIALS)
do
   echo "Start rendering trial ${i}..."
   .venv/bin/python3 render.py \
    --parent_folder ${PARENT_FOLDER} \
    --experiment_datetime ${EXPERIMENT_DATETIME} \
    --trial_id ${i}
done


