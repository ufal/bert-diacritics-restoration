import argparse
from collections import Counter

from utils_diacritization import get_aligned_tokens

from transformers import (
    AutoTokenizer,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", default='', type=str, help="")
    parser.add_argument("target_file", default='', type=str, help="")
    parser.add_argument("outfile", default="", help="")

    args = parser.parse_args()

    # Prepare task and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-multilingual-uncased',
        use_fast=True,
    )

    cnt = Counter()
    with open(args.input_file) as reader_input, open(args.target_file) as reader_target:
        for input_line, target_line in zip(reader_input, reader_target):
            input_line, target_line = input_line.strip(), target_line.strip()
            aligned_tokens = get_aligned_tokens(input_line, target_line, tokenizer)
            tokens_dia = [x[1] for x in aligned_tokens]
            cnt.update(tokens_dia)

    with open(args.outfile, 'w') as out_writer:
        for token, token_count in cnt.most_common():
            out_writer.write(f"{token}\t{token_count}\n")
