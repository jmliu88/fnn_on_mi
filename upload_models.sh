#!/bin/bash
models=~/experiments/models
while [ 1 -eq 1 ]; do 
	scp -rq $models james@115.159.58.218:~/models
	sleep 100
done
