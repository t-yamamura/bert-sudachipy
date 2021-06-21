import textspan
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from typing import List, Type

from sudachipy import dictionary


class CustomWordPieceTokenizer(BertWordPieceTokenizer):

    def __init__(self, **kwargs):
        super().__init__(handle_chinese_chars=False, strip_accents=False, **kwargs)

    def set_pre_tokenizer(self, custom_pre_tokenizer):
        self.pre_tokenizer = PreTokenizer.custom(custom_pre_tokenizer)

    def save(self, output_tokenizer_path, pretty=False):
        self.pre_tokenizer = BertPreTokenizer()
        super().save(output_tokenizer_path, pretty=pretty)

    def save_vocab(self, output_dir, prefix):
        self._tokenizer.model.save(output_dir, prefix)

    @staticmethod
    def load_from_tokenizer_file(input_tokenizer_path, pre_tokenizer):
        tokenizer = Tokenizer.from_file(input_tokenizer_path)
        tokenizer.pre_tokenizer = PreTokenizer.custom(pre_tokenizer)
        return tokenizer


class JapanesePreTokenizer:
    def custom_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        raise NotImplementedError()

    def pre_tokenize(self, pretok):
        pretok.split(self.custom_split)

    def normalized_string2spans(self, normalized_string: NormalizedString, tokens):
        tokens_spans = textspan.get_original_spans(tokens, str(normalized_string).strip())
        return [normalized_string[start:end] for char_spans in tokens_spans for start, end in char_spans]


class SudachipyPreTokenizer(JapanesePreTokenizer):
    def __init__(self, dict_type, mode):
        self.dict_type = dict_type
        self.mode = mode
        self.tokenizer_obj = dictionary.Dictionary(dict_type=dict_type).create()

    def tokenize(self, sequence: NormalizedString):
        surface_tokens = []
        normalized_tokens = []
        for m in self.tokenizer_obj.tokenize(str(sequence).strip(), self.mode):
            surface_tokens.append(m.surface())
            normalized_tokens.append((m.normalized_form()))
        return surface_tokens, normalized_tokens

    def custom_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        tokens, morphs = self.tokenize(normalized_string)
        spans = self.normalized_string2spans(normalized_string, tokens)
        if len(morphs) != len(spans):
            raise ValueError(len(morphs), len(spans), morphs, tokens, spans)
        _ = [span.replace(span.normalized, m) for m, span in zip(morphs, spans)]
        return spans
