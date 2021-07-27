#!/bin/bash

# Generate all required datasets

function clean_all() {
	# clean tessellation directory given as argument
	rm -rf "$1/gaps" "$1/gaps.csv" "$1/lines.csv"
	find "$1" -regex '.*png' -type f -delete
}

clean_all tessellations_train
./tessellation-generator/tessellator.py --params tessellations_train/params.json --outpath tessellations_train -n 2000 --seed 42

clean_all tessellations_test
./tessellation-generator/tessellator.py --params tessellations_test/params.json --outpath tessellations_test -n 1000 --seed 43

