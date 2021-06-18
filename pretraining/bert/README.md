# Sudachi BERT の学習

> 学習コーパス: wiki40b  
> https://www.tensorflow.org/datasets/catalog/wiki40b

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