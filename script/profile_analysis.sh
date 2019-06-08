#!/bin/bash

DATE=`date '+%Y-%m-%d_%H-%M-%S'`
CUR_DIR=`dirname "$(readlink -f "$0")"`
BASE_DIR=`readlink -f "$CUR_DIR/.."`
PROF_DIR=profile/analysis/$DATE
OUT_DIR=$BASE_DIR/$PROF_DIR
NVPROF=/usr/local/cuda-10.0/bin/nvprof

CMD="sudo $NVPROF --analysis-metrics --output-profile $OUT_DIR/profile.%p.nvprof $BASE_DIR/main $@"

echo
echo "=========================================================================="
echo "Start in-depth profiling and analysis of all kernels"
echo "Output directory:"
echo "$OUT_DIR"
echo "Command line:"
echo "$CMD"
echo "=========================================================================="
echo

mkdir -p $OUT_DIR
echo "$CMD" > $OUT_DIR/cmd.txt

$CMD

echo "=========================================================================="
echo
echo "To copy the profile on local machine, run on local machine from project base dir:"
echo "mkdir -p $PROF_DIR && gcloud compute scp project:$OUT_DIR/* $PROF_DIR/"
echo
