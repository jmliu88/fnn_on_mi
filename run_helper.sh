#!/bin/sh
python -u MI.py $1 | tee  `basename $1`.log
