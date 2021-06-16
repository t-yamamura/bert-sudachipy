#!/bin/sh

cd $(dirname $0) && cd ../../

WORK_DIR="pretraining/bert"

seq -f %02g 1 8|xargs -L 1 -I {} -P 8 python3 pretraining/bert/create_pretraining_data.py \
--input_file $WORK_DIR/datasets/corpus_splitted_by_paragraph/ja_wiki40b_small{}.paragraph \
--output_file $WORK_DIR/models/pretraining_small_{}.tfr_record \
--vocab_file $WORK_DIR/models/vocab.txt \
--tokenizer_type wordpiece \
--word_form_type normalized \
--split_mode C \
--sudachi_dic_type core \
--do_whole_word_mask \
--max_seq_length 512 \
--max_predictions_per_seq 80 \
--dupe_factor 10