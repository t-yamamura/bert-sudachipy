import argparse
import os
import sys
from glob import glob
from pathlib import Path
from sudachipy import tokenizer, dictionary
from tokenizers.normalizers import NFKC, Sequence
# from tokenizers.normalizers import Lowercase, NFKC, Strip, Sequence

from pretraining.bert.pre_tokenizers.pre_tokenizers import CustomWordPieceTokenizer, SudachipyPreTokenizer


def get_split_mode(split_mode: str):
    split_mode = split_mode.lower()
    if split_mode == 'a':
        return tokenizer.Tokenizer.SplitMode.A
    elif split_mode == 'b':
        return tokenizer.Tokenizer.SplitMode.B
    elif split_mode == 'c':
        return tokenizer.Tokenizer.SplitMode.C
    else:
        raise ValueError()


def main():
    args = get_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = glob(os.path.join(args.input_dir, '*.txt'))
    else:
        raise ValueError("`input_file` or `input_dir` must be specified.")

    print("input files")
    print("\n".join(files))

    settings = dict(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet
    )

    normalizer = Sequence([
        NFKC(),
    ])

    wp_tokenizer = CustomWordPieceTokenizer()
    wp_tokenizer.normalizer = normalizer

    split_mode = get_split_mode(args.split_mode)
    sudachi_pre_tokenizer = SudachipyPreTokenizer(args.dict_type, split_mode)
    wp_tokenizer.set_pre_tokenizer(sudachi_pre_tokenizer)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wp_tokenizer.train(files, **settings)

    print("#vocab:", wp_tokenizer.get_vocab_size())

    wp_tokenizer.save(os.path.join(args.output_dir, args.config_name))
    wp_tokenizer.save_vocab(args.output_dir, args.vocab_prefix)


def get_args():
    parser = argparse.ArgumentParser(description='train tokenizer')

    # input
    parser.add_argument('-f', '--input_file', default='',
                        help='input file to train tokenizer (corpus splitted by paragraph')
    parser.add_argument('-d', '--input_dir', default='',
                        help='input dir containing files to train tokenizer')

    # parameters
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--min_frequency', type=int, default=1)
    parser.add_argument('--limit_alphabet', type=int, default=5000)

    # sudachi
    parser.add_argument('--dict_type', default='core', choices=['small', 'core', 'full'])
    parser.add_argument('--split_mode', default='C', choices=['A', 'B', 'C', 'a', 'b', 'c'])

    # output
    parser.add_argument('-o', '--output_dir', help='path to be saved tokenizer file')
    parser.add_argument('-c', '--config_name', help='output json file name')
    parser.add_argument('-v', '--vocab_prefix', help='prefix of vocab file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
