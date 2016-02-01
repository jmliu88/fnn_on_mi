#!/bin/sh
for i in `find data/mat -name '*.mat'`; do

    echo $i
    OMP_NUM_THREADS=1 python -u MI.py $i | tee  `basename $i`.log &
done
