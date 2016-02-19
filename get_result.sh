#!/bin/sh
#find . -path './hidden*' -wholename '*csv' > results.txt
find . -path './PCA*' -name 'result*csv' > results.txt
#find . -path './hidden_100_rmsprop_stand_all_feat_*' -name 'result*' > results.txt
#find ./mean/ -name 'result*' > results.txt
python gether_result.py
echo "Write results to results.csv"
