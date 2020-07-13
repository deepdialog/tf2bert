
# BERT

Tensorflow 2.x

A lot of code is take from [bert for tf2](https://github.com/kpe/bert-for-tf2)

MIT LICENSE

```bash
$ python -m bert.tests.convert_official --input=bert_input --output=bert_output
```

```
mkdir -p ../bert_zip && for i in ./* ; do cd $i && tar czf ../../bert_zip/$i.tar.gz . && cd .. ; done;
```
