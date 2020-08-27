"""Test load official model, print difference."""

import os
import argparse
import json
import tensorflow as tf

from bert import BertModel, params


class BertToken2ids(tf.keras.models.Model):
    def __init__(self, word_index, **kwargs):
        super(BertToken2ids, self).__init__(**kwargs)
        self.construct(word_index)

    def construct(self, word_index):
        self.keys = tf.constant(list(word_index.keys()), dtype=tf.string)
        self.values = tf.constant(list(word_index.values()), dtype=tf.int32)
        self.table_init = tf.lookup.KeyValueTensorInitializer(
            self.keys,
            self.values)
        self.table = tf.lookup.StaticHashTable(
            self.table_init,
            tf.constant(word_index['[UNK]']))  # default value

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.string)],
        experimental_relax_shapes=True)
    def call(self, inputs):
        x = inputs
        x = self.table.lookup(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class BERT(tf.keras.Model):

    def __init__(self,
                 model_path,
                 word_index,
                 load_model,
                 num_hidden_layers=None, **kwargs):
        super(BERT, self).__init__(**kwargs)

        self.bert = load_model(model_path, num_hidden_layers=num_hidden_layers)
        self.tokenizer = BertToken2ids(word_index)
        self.make_type_ids = tf.keras.layers.Lambda(
            lambda x: tf.zeros(shape=tf.shape(x), dtype=tf.int32),
            name='make_type_ids')
        self.make_mask = tf.keras.layers.Lambda(
            # >= 0 的才是有效的
            # -1 是token2id的长度填充
            lambda x: tf.cast(tf.math.greater_equal(x, 0), tf.int32),
            name='make_mask')

    @tf.function
    def call_ids_mask_type(self, input_ids, input_mask, token_type_ids):
        output = self.bert({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'input_mask': input_mask
        })
        output_mask = tf.cast(input_mask, tf.float32)
        output_mask = tf.expand_dims(output_mask, -1)
        output['sequence_output'] = output['sequence_output'] * output_mask
        return output

    @tf.function
    def call_ids_mask(self, input_ids, input_mask):
        token_type_ids = self.make_type_ids(input_ids)
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_ids_type(self, input_ids, token_type_ids):
        input_mask = self.make_mask(input_ids)
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_ids(self, input_ids):
        token_type_ids = self.make_type_ids(input_ids)
        input_mask = self.make_mask(input_ids)
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_strs_mask_type(self, input_strs, input_mask, token_type_ids):
        input_ids = self.tokenizer(input_strs)
        input_ids = tf.math.abs(input_ids)  # 去掉-1的填充
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_strs_mask(self, input_strs, input_mask):
        input_ids = self.tokenizer(input_strs)
        token_type_ids = self.make_type_ids(input_ids)
        input_ids = tf.math.abs(input_ids)  # 去掉-1的填充
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_strs_type(self, input_strs, token_type_ids):
        input_ids = self.tokenizer(input_strs)
        input_mask = self.make_mask(input_ids)
        input_ids = tf.math.abs(input_ids)  # 去掉-1的填充
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call_strs(self, input_strs):
        input_ids = self.tokenizer(input_strs)
        token_type_ids = self.make_type_ids(input_ids)
        input_mask = self.make_mask(input_ids)
        input_ids = tf.math.abs(input_ids)  # 去掉-1的填充
        return self.call_ids_mask_type(
            input_ids, input_mask, token_type_ids)

    @tf.function
    def call(self, input_str):
        return self.call_strs(input_str)


def load_vocab(vocab_path):
    word_index = {}
    with open(vocab_path) as fp:
        for i, line in enumerate(fp):
            line = line.strip()  # .lower()
            word_index[line] = i
    word_index[''] = -1  # real padding
    # index_word = {v: k for k, v in word_index.items()}
    return word_index


def load_model(model_path, num_hidden_layers=None):
    ckpt_reader = tf.train.load_checkpoint(
        os.path.join(model_path, 'bert_model.ckpt'))
    config = json.load(open(os.path.join(model_path, 'bert_config.json')))

    loaded_params = {k: config[k] for k in params.keys()}
    # import pdb; pdb.set_trace()
    if num_hidden_layers is not None and num_hidden_layers > 0:
        loaded_params['num_hidden_layers'] = num_hidden_layers

    tfbert = BertModel(**loaded_params)

    tfbert_weights = {w.name: w for w in tfbert.weights}
    official_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    good = True
    for x in official_weights - set([x.split(':')[0]
                                     for x in tfbert_weights.keys()]):
        if 'adam' not in x and 'global_step' not in x:
            print('diff offi', x)
            good = False

    for x in set([x.split(':')[0]
                  for x in tfbert_weights.keys()]) - official_weights:
        if 'adam' not in x and 'global_step' not in x:
            print('diff ours', x)
            good = False

    assert good

    weight_tuples = []
    for k, v in tfbert_weights.items():
        name = k[:-2]
        if ckpt_reader.has_tensor(name):
            ckpt_value = ckpt_reader.get_tensor(name)
            weight_tuples.append((v, ckpt_value))
            assert v.shape == ckpt_value.shape, \
                f'{name} shape invalid {v.shape}, {ckpt_value.shape}'
        else:
            print(f'{name} weight not loaded')
    tf.keras.backend.batch_set_value(weight_tuples)
    return tfbert


def main():
    parser = argparse.ArgumentParser(
        description='convert official tf1 bert model to tf2')
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help='input tf1 bert dir')
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='output tf2 bert dir')
    parser.add_argument(
        '--num_hidden_layers',
        type=int,
        default=0,
        help='num_hidden_layers to keep, default all (0)'
    )
    args = parser.parse_args()

    model_path = args.input
    word_index = load_vocab(os.path.join(model_path, 'vocab.txt'))
    bert = BERT(model_path, word_index, load_model, args.num_hidden_layers)

    print('save to', model_path)

    strs = tf.TensorSpec(shape=[None, None],
                         dtype=tf.string,
                         name="input_strs")
    ids = tf.TensorSpec(shape=[None, None],
                        dtype=tf.int32,
                        name="input_ids")
    mask = tf.TensorSpec(shape=[None, None],
                         dtype=tf.int32,
                         name="input_mask")
    ttids = tf.TensorSpec(shape=[None, None],
                          dtype=tf.int32,
                          name="token_type_ids")

    tf.saved_model.save(bert, args.output, signatures={
        'serving_default': bert.call.get_concrete_function(
            strs
        ),

        'ids_mask_type': bert.call_ids_mask_type.get_concrete_function(
            ids, mask, ttids
        ),

        'ids_mask': bert.call_ids_mask.get_concrete_function(
            ids, mask
        ),

        'ids_type': bert.call_ids_type.get_concrete_function(
            ids, ttids
        ),

        'ids': bert.call_ids.get_concrete_function(
            ids
        ),

        'strs_mask_type': bert.call_strs_mask_type.get_concrete_function(
            strs, mask, ttids
        ),

        'strs_mask': bert.call_strs_mask.get_concrete_function(
            strs, mask
        ),

        'strs_type': bert.call_strs_type.get_concrete_function(
            strs, ttids
        ),

        'strs': bert.call_strs.get_concrete_function(
            strs
        ),
    })


if __name__ == "__main__":
    main()
