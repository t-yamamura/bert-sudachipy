# Sudachi BERT の学習

> 学習コーパス: wiki40b  
> https://www.tensorflow.org/datasets/catalog/wiki40b

```
# --recursive をつけて, tensorflow の models も一緒にclone
git clone --recursive https://github.com/t-yamamura/bert_sudachipy/
```


## 1.学習データの準備

wiki40bのダウンロードと事前学習用のためのデータの整形を行う．

ダウンロードしたデータは `./datasets/corpus` 以下に，
またそのコーパスをパラグラフ（段落）単位に空行を挟んだものは `./datasets/corpus_splitted_by_paragraph` 以下に出力される．

> ダウンロード時間は 90 * 3 min くらい？


```shell script
./run_prepare_dataset.sh
```


## 2.Tokenizer の学習

事前学習における Tokenizer の学習では，入力ファイルまたは入力ディレクトリを指定する．
入力ファイルのフォーマットは，[1. 学習データの準備](#1.学習データの準備) で生成された `./datasets/corpus_splitted_by_paragraph` 以下の段落区切りのファイルである．


```shell script
python pretraining/bert/train_pretokenizer.py
```

## 3. データの分割


```shell script
cd pretraining/bert
python data_split.py \
--input_file ./datasets/corpus_splitted_by_paragraph/ja_wiki40b_small.paragraph.txt \
--line_per_file 760000
```


## 4. 事前学習データの準備

```shell script
./run_create_pretraining_data.sh
```

## 5. 事前学習


```shell script
pwd
# /path/to/bert_sudachipy
sudo pip3 install -r models/official/requirements.txt
export PYTHONPATH="$PYTHONPATH:./models"
cd models/
WORK_DIR="../pretraining/bert"; py official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/models/pretraining_small_*record" \
--model_dir="$WORK_DIR/models/" \
--bert_config_file="$WORK_DIR/models/small_config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000

```


### 6. pytorch形式へのモデルの変換

```shell script
cd ../pretraining/bert/
python convert_original_tf2_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./models/ \
--config_file ./models/small_config.json \
--pytorch_dump_path ./models/pytorch_model.bin
```