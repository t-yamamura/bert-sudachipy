# bert-sudachipy


## 例: Bert用のTokenizerの使用方法

### 準備

```shell script
git clone https://github.com/t-yamamura/bert_sudachipy.git
export PYTHONPATH="$PYTHONPATH:`pwd`/bert_sudachipy"
```

### 実行例

```python
from transformers import BertConfig, BertForSequenceClassification
from bert_sudachipy.tokenization_bert_sudachipy import BertSudachipyTokenizer

tokenizer = BertSudachipyTokenizer.from_pretrained(
    '/path/to/vocab.txt',
    do_lower_case=False,
    word_tokenizer_type='sudachipy',
    subword_tokenizer_type='wordpiece'
)

wids = tokenizer.encode('毎月24日発売。', return_tensors='pt')
print(wids)
# >>> tensor([[   2, 6081, 3654,  982, 3891,   86,    3]])

print(tokenizer.convert_ids_to_tokens(wids[0].tolist()))
# >>> ['[CLS]', '毎月', '24', '日', '発売', '。', '[SEP]']

model_config = BertConfig.from_json_file('/path/to/model/config.json')
model = BertForSequenceClassification.from_pretrained(
    '/path/to/pytorch_model.bin',
    config=model_config
)

print(model(wids))
# >>> SequenceClassifierOutput(loss=None, logits=tensor([[-0.0339,  0.0467]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```
