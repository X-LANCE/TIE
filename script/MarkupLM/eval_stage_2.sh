#!/bin/bash

python -u -W ignore run.py                    \
  --train_file data/websrc1.0_train_.json     \
  --predict_file data/websrc1.0_dev_.json     \
  --root_dir data --do_eval                   \
  --model_type "$1" --model_name_or_path "$2" \
  --provided_tag_pred "$3"                    \
  --output_dir result/TIE_MarkopLM/stage_2/
