#!/bin/bash

# Define the range for gamma values
for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path meta-llama/Llama-3.1-8B-Instruct --model-id llama3-8B-experiments-redundant --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for gamma in {1..10}; do
  # Run the command 3 times for each gamma value
  for run in {1..2}; do
    echo "Running SPEC with gamma=$gamma, run=$run"
    start_time=$(date +%s)
    
    python fastchat/llm_judge/gen_model_answer.py --model-path meta-llama/Llama-3.1-8B-Instruct --model-id llama3-8B-experiments-redundant-copy --use-copy True --gamma $gamma --use-redundant True


    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
  done
done
