from .tokenization_bert_sudachipy import BertSudachipyTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# TODO: set official URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "megagonlabs/electra-base-ud-japanese": "https://.../vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "megagonlabs/electra-base-ud-japanese": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "megagonlabs/electra-base-ud-japanese": {
        "do_lower_case": False,
        "word_tokenizer_type": "sudachipy",
        "subword_tokenizer_type": "pos_substitution",
    },
}


class ElectraSudachipyTokenizer(BertSudachipyTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
