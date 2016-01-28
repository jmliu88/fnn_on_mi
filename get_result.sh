#!/bin/sh
find . *hidden*/result*.csv > results.txt
python gether_result.py
echo "Write results to results.csv"
