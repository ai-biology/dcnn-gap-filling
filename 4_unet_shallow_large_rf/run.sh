#!/usr/bin/bash

set -e

./train.py --datapath ../tessellations_train $@
./analyze.py --tessellations ../tessellations_test --results .
