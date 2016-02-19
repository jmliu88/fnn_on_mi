#!/bin/sh
for i in `find data/mat -name '*.mat'`; do

    echo $i
    OMP_NUM_THREADS=1 python -u MI_hyper.py $i | tee  `basename $i`_hyper.log &
    #OMP_NUM_THREADS=1 python -u MI_mean.py $i | tee  `basename $i`.log &
done
