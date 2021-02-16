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
""" Diacritization fine-tuning. """

import logging
import os
import pickle
import sys
import unicodedata
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
from dataclasses import dataclass
from filelock import FileLock
from torch import nn
from torch.utils.data.dataset import Dataset

import diacritization_stripping_data
from transformers import PreTrainedTokenizer

np.random.seed(42)


def strip_diacritics(text):
    d_map = diacritization_stripping_data.strip_diacritization_uninames
    output = ""
    for c in text:
        if c in d_map:
            output += d_map[c]
        else:
            output += c

    return output


logger = logging.getLogger(__name__)

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


@dataclass
class PredictInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DiacritizationDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            data_dir: str,
            input_train_file: str,
            target_train_file: str,
            input_dev_file: str,
            target_dev_file: str,
            input_test_file: str,
            target_test_file: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        '''

        :param lang: either concrete language (e.g. cs) or 'all' that means include all languages
        :param tokenizer:
        :param labels:
        :param model_type:
        :param max_seq_length:
        :param overwrite_cache:
        :param mode:
        '''

        self.label_map = {label: i for i, label in enumerate(labels)}
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        # Load data features from cache or dataset file
        cached_examples_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_examples_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_examples_file) and not overwrite_cache:
                logger.info(f"Loading examples from cached file {cached_examples_file}")
                with open(cached_examples_file, 'rb') as f:
                    self.examples = pickle.load(f)
            else:
                logger.info(f"Creating examples from dataset file at {data_dir}")

                if isinstance(mode, Split):
                    mode = mode.value

                logger.info(f"Processing {mode} data")

                if mode == 'test':
                    self.examples = read_examples_from_disk(input_test_file, target_test_file)
                elif mode == 'dev':
                    self.examples = read_examples_from_disk(input_dev_file, target_dev_file)
                elif mode == 'train':
                    self.examples = read_examples_from_disk(input_train_file, target_train_file)

                logger.info(f"Saving features into cached file {cached_examples_file}")

                with open(cached_examples_file, 'wb') as f:
                    pickle.dump(self.examples, f)

                    # torch.save(self.examples, cached_examples_file)
            self.random_examples_permutation = np.arange(len(self.examples))
            np.random.shuffle(self.random_examples_permutation)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> InputFeatures:
        # start_time = time.time()
        while True:
            try:
                example = self.examples[self.random_examples_permutation[i]]

                features = convert_example_to_features(
                    example,
                    self.label_map,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                    replace_with_copy_instruction=True,
                    use_instructions_as_labels=True
                )

                break

                # logging.info("time per item: {}".format(time.time() - start_time))
            except Exception as e:
                raise e
                # print("Trying another example i+1")
                # i += 1

        return features


class FileIteratorDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            file: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            labels
    ):
        '''

        :param lang: either concrete language (e.g. cs) or 'all' that means include all languages
        :param data_dir:
        :param tokenizer:
        :param labels:
        :param model_type:
        :param max_seq_length:
        :param overwrite_cache:
        :param mode:
        '''

        self.model_type = model_type
        self.tokenizer = tokenizer
        self.label_map = {label: i for i, label in enumerate(labels)}

        self.examples = []

        with open(file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.examples.append(line)

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        # start_time = time.time()
        example = self.examples[i]

        ## START PARAMS
        cls_token_at_end = bool(self.model_type in ["xlnet"])
        cls_token = self.tokenizer.cls_token
        cls_token_segment_id = 2 if self.model_type in ["xlnet"] else 0
        sep_token = self.tokenizer.sep_token
        sep_token_extra = False
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left = bool(self.tokenizer.padding_side == "left")
        pad_token = self.tokenizer.pad_token_id
        pad_token_segment_id = self.tokenizer.pad_token_type_id
        pad_token_label_id = self.pad_token_label_id
        sequence_a_segment_id = 0
        mask_padding_with_zero = True
        ## END PARAMS

        tokens = self.tokenizer.tokenize(example)
        unk_token = self.tokenizer.unk_token
        all_special_tokens_extended = self.tokenizer.all_special_tokens_extended

        # TODO remove this once training is corrected
        nohashes = []
        for token in tokens:
            if token in all_special_tokens_extended and token != unk_token:
                continue

            nohashes.append(token)

        tokens = nohashes

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if "token_type_ids" not in self.tokenizer.model_input_names:
            segment_ids = None

        # logging.info("time per item: {}".format(time.time() - start_time))
        return {"input_ids": torch.Tensor([input_ids]).long(),
                "attention_mask": torch.Tensor([input_mask]).long(),
                "token_type_ids": torch.Tensor([segment_ids]).long()}


def read_examples_from_disk(
        input_file: str,
        target_file: str,
        sentences_per_dataset=sys.maxsize
) -> List[Tuple[str, str]]:
    # start code

    guid_index = 1
    examples = []

    with open(input_file, encoding='utf-8') as inp_f, open(target_file,
                                                           encoding='utf-8') as tar_f:
        for line_index, (input_line, target_line) in enumerate(zip(inp_f, tar_f)):
            if line_index > sentences_per_dataset:
                break

            input_line, target_line = input_line.strip('\n'), target_line.strip('\n')
            if not input_line:
                continue

            # examples.append(InputExample(nodia=input_line, dia=target_line))
            examples.append(target_line)
            guid_index += 1

    return examples


def convert_example_to_features(
        example: Tuple[str, str],
        label_map,
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        replace_with_copy_instruction=True,
        use_instructions_as_labels=True
) -> InputFeatures:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    target_line = example
    input_line = strip_diacritics(target_line)

    aligned_tokens = get_aligned_tokens(input_line, target_line, tokenizer)
    tokens_dia_ids = []  # labels

    tokens_nodia = []
    for aligned_token in aligned_tokens:
        nodia_token, dia_token = aligned_token
        tokens_nodia.append(nodia_token)

        if not use_instructions_as_labels:
            if replace_with_copy_instruction and dia_token == nodia_token:
                tokens_dia_ids.append(label_map["<KEEP>"])
            elif nodia_token == tokenizer.unk_token:
                if replace_with_copy_instruction:
                    tokens_dia_ids.append(label_map["<KEEP>"])
                else:
                    tokens_dia_ids.append(label_map["<UNK>"])
            elif dia_token not in label_map:
                tokens_dia_ids.append(label_map["<UNK>"])
            else:
                tokens_dia_ids.append(label_map[dia_token])
        else:
            if dia_token == nodia_token:
                tokens_dia_ids.append(label_map["<KEEP>"])
            elif nodia_token == tokenizer.unk_token:
                tokens_dia_ids.append(label_map["<KEEP>"])
            else:
                dia_token_instructions = _text_to_dia_instructions(dia_token)
                if dia_token_instructions not in label_map:
                    tokens_dia_ids.append(label_map["<UNK>"])
                else:
                    tokens_dia_ids.append(label_map[dia_token_instructions])

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens_nodia) > max_seq_length - special_tokens_count:
        tokens_nodia = tokens_nodia[: (max_seq_length - special_tokens_count)]
        tokens_dia_ids = tokens_dia_ids[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens_nodia += [sep_token]
    tokens_dia_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens_nodia += [sep_token]
        tokens_dia_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens_nodia)

    if cls_token_at_end:
        tokens_nodia += [cls_token]
        tokens_dia_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens_nodia = [cls_token] + tokens_nodia
        tokens_dia_ids = [pad_token_label_id] + tokens_dia_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens_nodia)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        tokens_dia_ids = ([pad_token_label_id] * padding_length) + tokens_dia_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        tokens_dia_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(tokens_dia_ids) == max_seq_length

    # if ex_index < 5:
    #     logger.info("*** Example ***")
    #     # logger.info("guid: %s", example.guid)
    #     logger.info("tokens: %s", " ".join([str(x) for x in tokens_nodia]))
    #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    #     logger.info("label_ids: %s", " ".join([str(x) for x in tokens_dia_ids]))

    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

    return InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                         label_ids=tokens_dia_ids)


def get_aligned_tokens(input_line, target_line, tokenizer):
    unk_token = tokenizer.unk_token

    if not tokenizer.is_fast:
        raise ValueError("Do use fast tokenizer!")

    else:
        tokens_nodia_alignment = get_token_source_mapping(input_line, tokenizer)
        tokens_nodia = []
        tokens_dia = []

        for cur_tok, cur_tok_start, cur_tok_end in tokens_nodia_alignment:
            tokens_nodia.append(cur_tok)
            tokens_dia.append(target_line[cur_tok_start:cur_tok_end])

        aligned_tokens = []
        for token_nodia, token_dia in zip(tokens_nodia, tokens_dia):
            if token_nodia.startswith('##') and not token_dia.startswith('##'):
                token_dia = '##' + token_dia

            if token_nodia.startswith('▁') and not token_dia.startswith('▁'):
                token_dia = '▁' + token_dia

            if len(token_nodia) == len(token_dia):
                aligned_tokens.append([token_nodia, token_dia])

        return aligned_tokens


def normalize_text_from_tokenizer(text: str):
    # when constructing instruction for ##mne -> ##mně, ignore ## so that instruction is the same as for mne -> mně (less instructions)
    if text.startswith("##"):
        text = text[2:]

    if text.startswith('▁'):
        text = text[1:]

    return text


def dia_instructions_to_text(text: str, dia_instructions):
    if dia_instructions == '<KEEP>' or dia_instructions == '<UNK>':
        return text

    text = normalize_text_from_tokenizer(text)

    text = list(text)
    instructions = dia_instructions.split(';')
    for instruction in instructions:
        c_index, c_name_after_first_with = instruction.split(':')
        c_index = int(c_index)

        if c_index < len(text):
            c_name = unicodedata.name(text[c_index]) + " WITH " + c_name_after_first_with
            try:
                c_name = unicodedata.lookup(c_name)
                text[c_index] = c_name
            except KeyError:
                logging.info(f'{c_name} not found')
        else:
            logging.info(f'Skipping, because attempted {dia_instructions} on {text}, specifically {instruction}')

    return "".join(text)


def _text_to_dia_instructions(text: str):
    text = normalize_text_from_tokenizer(text)

    converted_label = []
    for c_index, c in enumerate(text):
        if strip_diacritics(c) != c:
            c_name = unicodedata.name(c)
            if 'WITH' in c_name:
                first_index_of_with = c_name.index("WITH")
                c_name_after_first_with = c_name[first_index_of_with + 4 + 1:]
                converted_label.append(f"{c_index}:{c_name_after_first_with}")

    if converted_label:
        return ";".join(converted_label)
    else:
        return "<KEEP>"


def get_token_source_mapping(input_line, tokenizer):
    unk_token = tokenizer.unk_token

    if not tokenizer.is_fast:
        raise ValueError('Not supported, must use Fast tokenizer!')
    else:
        input_line_encoded = tokenizer(input_line)
        input_line_tokens = tokenizer.convert_ids_to_tokens(input_line_encoded['input_ids'])
        input_line_tokens_special_out = []
        for input_line_token in input_line_tokens:
            # skip all special tokens (e.g. [CLS], [PAD]) - these are added later, but not unknown token
            if input_line_token in tokenizer.all_special_tokens_extended and input_line_token != unk_token:
                continue

            input_line_tokens_special_out.append(input_line_token)
        input_line_tokens = input_line_tokens_special_out

        input_line_char_to_token_indices = [input_line_encoded.char_to_token(i) for i in range(len(input_line))]
        aligned_tokens = [[x, None, None] for x in input_line_tokens]
        for i in range(len(input_line)):
            cur_char_word_ind = input_line_char_to_token_indices[i]

            if cur_char_word_ind is None:
                continue

            cur_char_word_ind = cur_char_word_ind - 1  # char_to_token indexes from 1
            if aligned_tokens[cur_char_word_ind][1] is None:
                aligned_tokens[cur_char_word_ind][1] = i

            aligned_tokens[cur_char_word_ind][2] = i + 1

        for i in range(len(aligned_tokens)):
            if i > 0 and aligned_tokens[i][1] is None:
                aligned_tokens[i][1] = aligned_tokens[i - 1][1]
                aligned_tokens[i][2] = aligned_tokens[i - 1][2]

        if len(aligned_tokens) != len(input_line_tokens):
            print(input_line)
            print(aligned_tokens)
            print(input_line_tokens)
            raise ValueError()
    return aligned_tokens


def get_labels(path: str) -> List[str]:
    # now only instructions are supported
    labels = dict()
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            dia_token, count = line.split('\t')
            if len(dia_token) > 2 and (dia_token[0] == dia_token[-1] == "\"" or dia_token[0] == dia_token[-1] == "'"):
                dia_token = dia_token[1:-1]

            if strip_diacritics(dia_token) != dia_token:
                converted_label = _text_to_dia_instructions(dia_token)

                if converted_label not in labels:
                    labels[converted_label] = 0

                labels[converted_label] += int(count)
            else:
                if '<KEEP>' not in labels:
                    labels['<KEEP>'] = 0
                labels['<KEEP>'] += int(count)

    print(f'Original instruction count {len(labels)}')
    # filter out labels that occured only once (CommonCrawl outliers)
    filtered_labels = []
    for label, label_count in labels.items():
        if label_count >= 2:
            filtered_labels.append(label)

    print('Filtering out instructions that occurred only once....')
    print(
        f'New instruction set size: {len(filtered_labels)}. That is, we filtered out {len(labels) - len(filtered_labels)}')
    # append UNK
    labels = ["<UNK>", "<KEEP>"] + filtered_labels
    return labels
