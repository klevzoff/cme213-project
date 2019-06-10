#!/bin/bash

DATE=`date '+%Y-%m-%d_%H-%M-%S'`
CUR_DIR=`dirname "$(readlink -f "$0")"`
BASE_DIR=`readlink -f "$CUR_DIR/.."`
PROF_DIR=profile/mpi/$DATE
OUT_DIR=$BASE_DIR/$PROF_DIR
NVPROF=/usr/local/cuda-10.0/bin/nvprof
MPIRUN="mpirun -np 4"

CMD="$MPIRUN sudo -E $NVPROF --output-profile $OUT_DIR/profile.%p.nvprof $BASE_DIR/main $@"

echo
echo "==================================="
echo "Start profiling with MPI on 4 ranks"
echo "Output directory:"
echo "$OUT_DIR"
echo "Command line:"
echo "$CMD"
echo "==================================="
echo

mkdir -p $OUT_DIR
echo "$CMD" > $OUT_DIR/cmd.txt

$CMD

echo "=========================================================================="
echo
echo "To copy the profile on local machine, run on local machine from project base dir:"
echo "mkdir -p $PROF_DIR && gcloud compute scp project:$OUT_DIR/* $PROF_DIR/"
echo
