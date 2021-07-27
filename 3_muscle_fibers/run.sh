#!/usr/bin/bash

set -e

SAMPLES="/nfs/numerik/bzfpeppe/data/muscle-fibers/samples_split.tsv"
EXPERIMENT="train_depth"

RUN_IDS=$(seq 1 10)
for RUN_ID in "$RUN_IDS"; do
    ./train.py -s "$SAMPLES" -p "params.json" -e "$EXPERIMENT" -r "$RUN_ID"
# ./analyze.py -p "params.json" -e "$EXPERIMENT" -s "$SAMPLES" -u "unets-$EXPERIMENT" --no-precrop
# ./plot.py -a "unets-${EXPERIMENT}/analysis.pickle" -e "$EXPERIMENT" -o "plots-$EXPERIMENT"