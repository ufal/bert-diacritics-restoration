#!/usr/bin/env bash

set -ex 

# TODO start-fill-in-with-your-data
# fill in the paths to the input(no diacritics)-target (with diacritics) files
INPUT_TRAIN_FILE=""
TARGET_TRAIN_FILE=""
INPUT_DEV_FILE=""
TARGET_DEV_FILE=""
INPUT_TEST_FILE=""
TARGET_TEST_FILE=""

# fill in the path to SUBWORD FREQUENCIES obtained with generate_subword_frequencies.py script
LABELS=""

# path to store model checkpoints
OUTPUT_DIR=""

# path to store generated data and intermediate cache files
DATA_DIR=""
# end-TODO start-fill-in-with-your-data

# other hyperparameters
NUM_EPOCHS=10
BERT_MODEL=bert-base-multilingual-uncased
TOKENIZER_NAME=bert-base-multilingual-uncased
MAX_LENGTH=128
BATCH_SIZE=64
GRAD_ACC_STEPS=32
SAVE_STEPS=400
SEED=1

mkdir -p $OUTPUT_DIR
mkdir -p $DATA_DIR

# run training
python run_diacritization.py \
 --data_dir $DATA_DIR \ 
 --labels $LABELS \
 --model_name_or_path $BERT_MODEL \ 
 --tokenizer_name $TOKENIZER_NAME \
 --output_dir $OUTPUT_DIR \
 --max_seq_length $MAX_LENGTH \ 
 --num_train_epochs $NUM_EPOCHS \ 
 --per_device_train_batch_size $BATCH_SIZE \
 --per_device_eval_batch_size 1 \ 
 --gradient_accumulation_steps $GRAD_ACC_STEPS \
 --cache_dir $DATA_DIR \
 --save_steps $SAVE_STEPS \ 
 --seed $SEED \ 
 --do_train \ 
 --overwrite_output_dir \
 --input_train_file $INPUT_TRAIN_FILE \
 --target_train_file $TARGET_TRAIN_FILE \
 --input_dev_file $INPUT_DEV_FILE \
 --target_dev_file $TARGET_DEV_FILE \
 --input_test_file $INPUT_TEST_FILE \
 --target_test_file TARGET_TEST_FILE
