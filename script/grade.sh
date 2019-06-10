#!/bin/bash

EXE="$(dirname "$0")/../main"
MPIRUN="mpirun -np 4"

if [ -z "$1" ]; then
    CMD="$MPIRUN $EXE"
else
    case $1 in
        1) N=100 R=0.0001 L=0.001 E=40 B=800 ;;
        2) N=100 R=0.0001 L=0.010 E=10 B=800 ;;
        3) N=100 R=0.0001 L=0.025 E=1  B=800 ;;
        4) G=4 ;;
        *) echo "Invalid mode: $1"; exit 1 ;;
    esac
    if [ -z "$G" ]; then
        CMD="$MPIRUN $EXE -n $N -r $R -l $L -e $E -b $B"
    else
        CMD="$EXE -g $G"
    fi
fi

echo
echo "GRADE MODE: $1"
echo "$CMD"
echo "===================================================================="
$CMD
echo "===================================================================="
echo
