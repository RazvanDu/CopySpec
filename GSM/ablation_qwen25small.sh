#!/bin/bash

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python cp_gsm_vikas.py --model-path "Qwen/Qwen2.5-7B-Instruct" --use-copy True --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python cp_gsm_vikas.py --model-path "Qwen/Qwen2.5-7B-Instruct"

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done