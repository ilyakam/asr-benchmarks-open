#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("tiny.en" "small.en" "base.en" "medium.en" "large-v1" "large-v2" "large-v3" "large-v3-turbo")

# Datasets and their splits (format: "dataset_path:dataset_name:split")
DATASETS_AND_SPLITS=(
    "hf-audio/esb-datasets-test-only-sorted:ami:test"
    "hf-audio/esb-datasets-test-only-sorted:common_voice:test"
    "hf-audio/esb-datasets-test-only-sorted:earnings22:test"
    "hf-audio/esb-datasets-test-only-sorted:gigaspeech:test"
    "hf-audio/esb-datasets-test-only-sorted:librispeech:test.clean"
    "hf-audio/esb-datasets-test-only-sorted:librispeech:test.other"
    "hf-audio/esb-datasets-test-only-sorted:spgispeech:test"
    "hf-audio/esb-datasets-test-only-sorted:tedlium:test"
    "hf-audio/esb-datasets-test-only-sorted:voxpopuli:test"
    "ilyakam/librispeech-long:librispeech_long:test_clean"
    "ilyakam/librispeech-long:librispeech_long:test_other"
)

# Configurable variables with defaults
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE_ID=${DEVICE_ID:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:--1}

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ )); do
    MODEL_ID=${MODEL_IDs[$i]}

    for ds_split in "${DATASETS_AND_SPLITS[@]}"; do
        IFS=':' read -r dataset_path dataset split <<< "$ds_split"

        python run_eval.py \
            --model_id="${MODEL_ID}" \
            --dataset_path="${dataset_path}" \
            --dataset="${dataset}" \
            --split="${split}" \
            --device="${DEVICE_ID}" \
            --batch_size="${BATCH_SIZE}" \
            --max_eval_samples="${MAX_EVAL_SAMPLES}"
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
