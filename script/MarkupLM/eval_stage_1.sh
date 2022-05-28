#!/bin/bash

python -u -W ignore src/run.py                  \
  --train_file data/websrc1.0_train_.json       \
  --predict_file data/websrc1.0_dev_.json       \
  --root_dir data --do_eval                     \
  --model_type markuplm                         \
  --model_name_or_path microsoft/markuplm-large \
  --output_dir result/TIE_MarkupLM/             \
  --eval_all_checkpoints