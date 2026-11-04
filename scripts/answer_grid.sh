#!/usr/bin/env bash
# Answer runs on one split: closed book, then one shot and agent for each retriever and model.
# Each run has its own spending cap; runs resume from their JSONL if interrupted.
set -euo pipefail
split=${1:-test}
retrievers=(bm25 dense:BAAI/bge-m3 dense:models/bge-m3-ft)

for model in qwen haiku; do
  lexsearch answer closed_book --model $model --split $split --max-cost 1
done
for r in "${retrievers[@]}"; do
  for model in qwen haiku; do
    lexsearch answer one_shot --model $model --retriever $r --split $split --max-cost 2
  done
done
for r in "${retrievers[@]}"; do
  lexsearch answer agent --model qwen --retriever $r --split $split --max-cost 3
  lexsearch answer agent --model haiku --retriever $r --split $split --max-cost 6
done
