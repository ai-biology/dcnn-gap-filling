#!/usr/bin/bash

set -e

OUTPATH=./results
mkdir -p "$OUTPATH"

RUN_IDS=$(seq 1 3)
for RUN_ID in $RUN_IDS; do
    TRAINSET_PATH="${OUTPATH}/${RUN_ID}/tessellations_train"
    TESTSET_PATH="${OUTPATH}/${RUN_ID}/tessellations_test"
    RESULTS_PATH="${OUTPATH}/${RUN_ID}/results"

    mkdir -p "$TRAINSET_PATH"
    mkdir -p "$TESTSET_PATH"
    mkdir -p "$RESULTS_PATH"

    echo "[Run $RUN_ID] Generate training data in $TRAINSET_PATH"
    train_seed=$(( $RUN_ID + 42 ))
    ./tessellation-generator/tessellator.py \
        --params ../tessellations_train/params.json \
        --outpath "$TRAINSET_PATH" \
        -n 2000 \
        --seed "$train_seed"

    echo "[Run $RUN_ID] Generate test data in $TESTSET_PATH"
    test_seed=$(( $RUN_ID + 442 ))
    ./tessellation-generator/tessellator.py \
        --params ../tessellations_test/params.json \
        --outpath "$TESTSET_PATH" \
        -n 1000 \
        --seed "$test_seed"

    cp ../tessellations_train/params.json $TRAINSET_PATH/params.json
    cp ../tessellations_test/params.json $TESTSET_PATH/params.json

    echo "[Run $RUN_ID] Train U-Nets, save to $RESULTS_PATH"
    ./train.py --datapath "$TRAINSET_PATH" --outpath "$RESULTS_PATH"

    echo "[Run $RUN_ID] Analyze U-Nets, save to $RESULTS_PATH"
    ./analyze.py --tessellations "$TESTSET_PATH" --results "$RESULTS_PATH"
done

echo "Generate aggregate plots"
./plot.py -t ../tessellations_test -r results
