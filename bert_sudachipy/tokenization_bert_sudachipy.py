from collections import OrderedDict
import copy
import os
from typing import Dict, Optional, Tuple

from transformers.models.bert_japanese.tokenization_bert_japanese import CharacterTokenizer
from transformers.models.bert.tokenization_bert import WordpieceTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from .sudachipy_word_tokenizer import SudachipyWordTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocabulary(vocab_file = VOCAB_FILES_NAMES["vocab_file"]):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def save_vocabulary(vocab: Dict[str, int], save_directory: str, filename_prefix: Optional[str] = None, vocab_file: str = VOCAB_FILES_NAMES["vocab_file"]) -> Tuple[str]:
    index = 0
    if os.path.isdir(save_directory):
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + vocab_file
        )
    else:
        vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            assert index == token_index, f"Error {vocab_file}: vocabulary indices are not consecutive, '{token}' {index} != {token_index}."
            writer.write(token + "\n")
            index += 1
    return (vocab_file,)


SUBWORD_TOKENIZER_TYPES = [
    "pos_substitution",
    "wordpiece",
    "character",
]

WORD_FORM_TYPES = {
    "surface": lambda m: m.surface(),
    "dictionary": lambda m: m.dictionary_form(),
    "normalized": lambda m: m.normalized_form(),
    "dictionary_and_surface": lambda m: m.surface() if m.part_of_speech()[0] in CONJUGATIVE_POS else m.dictionary_form(),
    "normalized_and_surface": lambda m: m.surface() if m.part_of_speech()[0] in CONJUGATIVE_POS else m.normalized_form(),
}

CONJUGATIVE_POS = {'動詞', '形容詞', '形容動詞', '助動詞'}


def pos_subsutitution_format(token):
    hierarchy = token.part_of_speech()
    pos = f"[{hierarchy[0]}"
    for p in hierarchy[1:]:
        if p == "*":
            break
        pos += "-" + p
    return pos + "]"


class BertSudachipyTokenizer(PreTrainedTokenizer):

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_subword_tokenize=True,
            subword_tokenizer_type="pos_substitution",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            word_form_type="surface",
            sudachipy_kwargs=None,
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_subword_tokenize=do_subword_tokenize,
            subword_tokenizer_type=subword_tokenizer_type,
            word_form_type=word_form_type,
            sudachipy_kwargs=sudachipy_kwargs,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        self.vocab = load_vocabulary(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.sudachipy_kwargs = copy.deepcopy(sudachipy_kwargs)

        self.word_tokenizer = SudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))
        self.word_form_type = word_form_type

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "pos_substitution":
                self.subword_tokenizer = None
            elif subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # TODO: need to investigate the serialization behavior
    def __getstate__(self):
        state = dict(self.__dict__)
        del state["word_tokenizer"]
        return state

    # TODO: need to investigate the serialization behavior
    def __setstate__(self, state):
        self.__dict__ = state
        self.word_tokenizer = SudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))

    def _tokenize(self, text, **kwargs):
        tokens = self.word_tokenizer.tokenize(text)
        word_format = WORD_FORM_TYPES[self.word_form_type]
        if self.do_subword_tokenize:
            if self.subword_tokenizer_type == "pos_substitution":
                def _substitution(token):
                    word = word_format(token)
                    if word in self.vocab:
                        return word
                    substitute = pos_subsutitution_format(token)
                    if substitute in self.vocab:
                        return substitute
                    return self.unk_token
                split_tokens = [_substitution(token) for token in tokens]
            else:
                split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(
                    word_format(token)
                )]
        else:
            split_tokens = [word_format(token) for token in tokens]

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory, filename_prefix=None, vocab_file_name=VOCAB_FILES_NAMES["vocab_file"]):
        return save_vocabulary(self.vocab, save_directory, filename_prefix, vocab_file_name)
