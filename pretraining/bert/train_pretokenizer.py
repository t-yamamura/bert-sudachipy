import os
import sys
from pathlib import Path
from sudachipy import tokenizer, dictionary
from tokenizers.normalizers import NFKC, Sequence
# from tokenizers.normalizers import Lowercase, NFKC, Strip, Sequence

from pretraining.bert.pre_tokenizers.pre_tokenizers import CustomWordPieceTokenizer, SudachipyPreTokenizer


def main():
    files = [Path('./datasets/corpus_splitted_by_paragraph/ja_wiki40b_small.paragraph.txt').as_posix()]
    print(files)

    settings = dict(
        vocab_size=10000,
        min_frequency=1,
        limit_alphabet=5000,
        show_progress=True,
    )

    normalizer = Sequence([
        NFKC(),
    ])

    wp_tokenizer = CustomWordPieceTokenizer()
    wp_tokenizer.normalizer = normalizer

    mode = tokenizer.Tokenizer.SplitMode.C
    sudachi_pre_tokenizer = SudachipyPreTokenizer("core", mode)
    wp_tokenizer.set_pre_tokenizer(sudachi_pre_tokenizer)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wp_tokenizer.train(files, **settings)

    print("#vocab:", wp_tokenizer.get_vocab_size())

    wp_tokenizer.save("./models/tokenizer.json")
    wp_tokenizer.save_vocab('./models/', 'sudachi_wordpiece')


if __name__ == '__main__':
    main()
