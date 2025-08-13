#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

#considering latest model
MODEL_IDs=("nvidia/parakeet-tdt-0.6b-v2")
# For previous parakeet models: FC-L, FC-XL, FC-XXL, C-L and C-S RNNT models
#  ("nvidia/parakeet-tdt-1.1b" "nvidia/parakeet-rnnt-1.1b" "nvidia/parakeet-rnnt-0.6b" "nvidia/stt_en_fastconformer_transducer_large" "nvidia/stt_en_conformer_transducer_large" "stt_en_conformer_transducer_small")

# Datasets and their splits (format: "dataset:split")
DATASETS_AND_SPLITS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "tedlium:test"
    "voxpopuli:test"
)

# Configurable variables with defaults
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE_ID=${DEVICE_ID:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:--1}

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ )); do
    MODEL_ID=${MODEL_IDs[$i]}

    for ds_split in "${DATASETS_AND_SPLITS[@]}"; do
        dataset="${ds_split%%:*}"
        split="${ds_split#*:}"

        python run_eval.py \
            --model_id="${MODEL_ID}" \
            --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
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
