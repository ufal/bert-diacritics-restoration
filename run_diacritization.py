# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """

import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from utils_diacritization import DiacritizationDataset, Split, get_labels, FileIteratorDataset, \
    dia_instructions_to_text, get_token_source_mapping

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir.."}
    )

    input_train_file: str = field(
        metadata={"help": "."}
    )
    target_train_file: str = field(
        metadata={"help": "."}
    )
    input_dev_file: str = field(
        metadata={"help": "."}
    )
    target_dev_file: str = field(
        metadata={"help": "."}
    )
    input_test_file: str = field(
        metadata={"help": "."}
    )
    target_test_file: str = field(
        metadata={"help": "."}
    )

    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels (target vocabulary)."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    prediction_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels (target vocabulary)."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
                        os.path.exists(training_args.output_dir)
                    and os.listdir(training_args.output_dir)
                and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare task and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise ValueError("Should have used fast tokenizer, but it is not available")

    # Generate target instruction set
    cached_labels_file = os.path.join(
        data_args.data_dir, "cached_{}_{}.labels".format(tokenizer.__class__.__name__, str(data_args.max_seq_length)),
    )

    logging.info(f'Looking for cached labels file in {cached_labels_file}')
    if os.path.exists(cached_labels_file) and not data_args.overwrite_cache:
        labels = torch.load(cached_labels_file)
        logging.info('Loaded cached labels.')
    else:
        labels = get_labels(data_args.labels)
        torch.save(labels, cached_labels_file)
        logging.info('Created new labels.')

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    logging.info(f"Output vocabulary has size of {num_labels}")

    # Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        DiacritizationDataset(
            input_train_file=data_args.input_train_file,
            target_train_file=data_args.target_train_file,
            input_dev_file=data_args.input_dev_file,
            target_dev_file=data_args.target_dev_file,
            input_test_file=data_args.input_test_file,
            target_test_file=data_args.target_test_file,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        DiacritizationDataset(
            input_train_file=data_args.input_train_file,
            target_train_file=data_args.target_train_file,
            input_dev_file=data_args.input_dev_file,
            target_dev_file=data_args.target_dev_file,
            input_test_file=data_args.input_test_file,
            target_test_file=data_args.target_test_file,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:

        test_dataset = FileIteratorDataset(
            file=data_args.prediction_file_path,
            tokenizer=tokenizer,
            model_type=model_args.model_name_or_path,
            labels=labels
        )

        logging.info(f'Test dataset created and has {len(test_dataset)} elements.')

        label_map = {i: label for i, label in enumerate(labels)}

        preds = []
        num_runtime_error_sentences = 0
        for test_input_i, test_input in enumerate(test_dataset):
            # if len(test_input) > 512:
            # max number of input subwords - we need to chunk it
            preds_cur = [[None]]
            try:
                _, logits, _ = trainer.prediction_step(model, test_input, False)
                predictions = logits.cpu()
                # predictions, label_ids, metrics = trainer.predict(test_dataset)
                preds_cur = np.argmax(predictions, axis=2)
            except RuntimeError as e:
                logging.info("Too long sentence.")
                logging.info(test_input_i)
                num_runtime_error_sentences += 1

            preds.extend(preds_cur)
            if test_input_i % 100 == 0:
                logging.info(f'Predicted for single sentence {test_input_i}')

        logging.info(f'Dataset translated and has {len(preds)} elements')
        logging.info(f'Num runtime errors: {num_runtime_error_sentences}')

        with open(data_args.prediction_file_path, encoding='utf-8') as f:
            for line, pred in zip(f, preds):
                line = line.strip()

                if len(pred) == 1 and pred[0] is None:  # too long sentence, need to copy
                    print(line)
                    continue

                token_source_mapping = get_token_source_mapping(line, tokenizer)
                tokens = ['CLS'] + token_source_mapping + ['PAD']
                new_sentence = list(line[:])
                # logging.info(line)
                for token_ind, (token, p) in enumerate(zip(tokens, pred)):
                    if token_ind == 0 or token_ind == len(tokens) - 1:
                        continue

                    token_start = token[1]
                    token_end = token[2]
                    token = line[token_start:token_end]

                    p = p.item()
                    label = label_map[p]

                    # logging.info(label)
                    if token == '<UNK>' or token == '[UNK]':
                        continue  # just copy source (do nothing)
                    else:
                        converted_subword = dia_instructions_to_text(token, label)
                        new_sentence[token_start:token_end] = list(converted_subword)

                new_sentence = "".join(new_sentence)

                print(new_sentence)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
