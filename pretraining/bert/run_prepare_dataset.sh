#!/bin/bash

cd $(dirname $0)

DATASET_DIR="./datasets"
OUTPUT_DIR="${DATASET_DIR}/corpus_splitted_by_paragraph"

# download dataset
mkdir -p ${DATASET_DIR}
for target in "validation" "test"; do
  time python3 prepare_dataset.py --target ${target} > ${DATASET_DIR}/ja_wiki40b_${target}.txt
done

### split dataset for each paragraph

#for target in "train" "validation" "test"; do
mkdir -p ${OUTPUT_DIR}
#for target in "small"; do
for target in "train" "validation" "test"; do
  cat ${DATASET_DIR}/ja_wiki40b_${target}.txt | sed -e "s/_START_ARTICLE_//g" -e "s/_START_PARAGRAPH_//g" | cat -s > ${OUTPUT_DIR}/ja_wiki40b_${target}.paragraph.txt
done

