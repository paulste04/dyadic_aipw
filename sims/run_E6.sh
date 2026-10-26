#!/bin/bash
# Runs the ML / cross-fitting experiment in resumable chunks.  Safe to re-launch: finished
# replications are skipped.  LightGBM runs single-threaded; parallelism is across replications.
cd "$(dirname "$0")"
python3 run.py E6 --family gauss   --N 300 >> results/log_E6.txt 2>&1
python3 run.py E6 --family poisson --N 300 >> results/log_E6.txt 2>&1
python3 run.py E6 --family gauss   --N 600 >> results/log_E6.txt 2>&1
python3 run.py E6 --family poisson --N 600 >> results/log_E6.txt 2>&1
echo ALLDONE >> results/log_E6.txt
