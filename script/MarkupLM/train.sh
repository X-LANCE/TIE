#!/bin/bash

python -u -W ignore src/run.py                  \
  --train_file data/websrc1.0_train_.json       \
  --predict_file data/websrc1.0_dev_.json       \
  --root_dir data --separate_read 8             \
  --model_type markuplm --do_train              \
  --model_name_or_path microsoft/markuplm-large \
  --output_dir result/TIE_MarkupLM/
