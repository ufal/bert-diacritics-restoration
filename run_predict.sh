#!/usr/bin/env bash

set -ex

IN_FILE=$1
OUT_FILE=$2
MODEL=$3
DATA_DIR=$4

tmpfile=$(mktemp /tmp/abc-script.XXXXXX)

TOKENIZER_NAME=bert-base-multilingual-uncased
MAX_LENGTH=128
BERT_MODEL=bert-base-multilingual-uncased

python run_diacritization.py \
 --data_dir $DATA_DIR \
 --model_name_or_path $MODEL \
 --tokenizer_name $TOKENIZER_NAME \
 --output_dir $MODEL \
 --per_device_eval_batch_size 1 \
 --max_seq_length $MAX_LENGTH \
 --cache_dir $DATA_DIR \
 --do_predict \
 --input_train_file $tmpfile \
 --target_train_file $tmpfile \
 --input_dev_file $tmpfile \
 --target_dev_file $tmpfile \
 --input_test_file $tmpfile \
 --target_test_file $tmpfile \
 --prediction_file_path $IN_FILE > $OUT_FILE

rm $tmpfile
