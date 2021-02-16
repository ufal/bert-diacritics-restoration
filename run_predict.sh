#!/usr/bin/env bash

set -ex

IN_FILE=$1
OUT_FILE=$2
MODEL=$3
DATA_DIR=$4


TOKENIZER_NAME=bert-base-multilingual-uncased
MAX_LENGTH=128
BERT_MODEL=bert-base-multilingual-uncased

python run_diacritization.py \
 --data_dir $DATA_DIR \
 --model_name_or_path $MODEL \
 --tokenizer_name $TOKENIZER_NAME \
 --use_fast \
 --output_dir $MODEL \
 --per_device_eval_batch_size 1 \
 --max_seq_length $MAX_LENGTH \
 --cache_dir $DATA_DIR \
 --do_predict \
 --prediction_file_path $IN_FILE > $OUT_FILE
