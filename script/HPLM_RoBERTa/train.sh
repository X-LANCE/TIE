#!/bin/bash

python -u -W ignore src/run.py            \
  --train_file data/websrc1.0_train_.json \
  --predict_file data/websrc1.0_dev_.json \
  --root_dir data --separate_read 8       \
  --model_type roberta --do_train         \
  --model_name_or_path roberta-large      \
  --output_dir result/TIE_HPLM_RoBERTa/
