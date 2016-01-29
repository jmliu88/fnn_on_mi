#!/bin/sh
find . -wholename *hidden*/result*.csv > results.txt
python gether_result.py
echo "Write results to results.csv"
