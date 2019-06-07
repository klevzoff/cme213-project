#!/bin/bash

DATE=`date '+%Y-%m-%d_%H-%M-%S'`
CUR_DIR=`dirname "$(readlink -f "$0")"`
BASE_DIR=`readlink -f "$CUR_DIR/.."`
OUT_DIR=$BASE_DIR/profile/serial/$DATE
NVPROF=/usr/local/cuda-10.0/bin/nvprof

CMD="sudo $NVPROF --output-profile $OUT_DIR/profile.%p.nvprof $BASE_DIR/main $@"

echo
echo "==================================="
echo "Start serial run profiling"
echo "Output directory:"
echo "$OUT_DIR"
echo "Command line:"
echo "$CMD"
echo "==================================="
echo

mkdir -p $OUT_DIR
echo "$CMD" > $OUT_DIR/cmd.txt

$CMD
