"""Test load official model, print difference."""

import os
import argparse
import json
import tensorflow as tf

from bert import BertModel, albert_params as params


class BertToken2ids(tf.keras.models.Model):
    def __init__(self, word_index, **kwargs):
        super(BertToken2ids, self).__init__(**kwargs)
        self.construct(word_index)

    def construct(self, word_index):
        self.keys = tf.constant(list(word_index.keys()), dtype=tf.string)
        self.values = tf.constant(list(word_index.values()), dtype=tf.int32)
        self.table_init = tf.lookup.KeyValueTensorInitializer(
            self.keys, self.values)
        self.table = tf.lookup.StaticHashTable(
            self.table_init, tf.constant(word_index['[UNK]']))  # default value

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
    def __init__(self, model_path, **kwargs):
        super(BERT, self).__init__(**kwargs)

        self.bert = load_model(model_path)
        word_index = load_vocab(os.path.join(model_path, 'vocab_chinese.txt'))
        self.tokenizer = BertToken2ids(word_index)

        self.make_type_ids = tf.keras.layers.Lambda(
            lambda x: tf.zeros(shape=tf.shape(x), dtype=tf.int32),
            name='make_type_ids')
        self.make_mask = tf.keras.layers.Lambda(
            # >= 0 的才是有效的
            # -1 是token2id的长度填充
            lambda x: tf.cast(tf.math.greater_equal(x, 0), tf.int32),
            name='make_mask')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.string)],
        experimental_relax_shapes=True)
    def call(self, input_str):
        input_ids = self.tokenizer(input_str)
        segment_ids = self.make_type_ids(input_ids)
        input_mask = self.make_mask(input_ids)
        input_ids = tf.math.abs(input_ids)  # 去掉-1的填充
        output = self.bert({
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask
        })
        output_mask = tf.cast(input_mask, tf.float32)
        output_mask = tf.expand_dims(output_mask, -1)
        output['sequence_output'] = output['sequence_output'] * output_mask
        return output


def load_vocab(vocab_path):
    word_index = {}
    with open(vocab_path) as fp:
        for i, line in enumerate(fp):
            line = line.strip()  # .lower()
            word_index[line] = i
    word_index[''] = -1  # real padding
    # index_word = {v: k for k, v in word_index.items()}
    return word_index


def load_model(model_path):
    ckpt_reader = tf.train.load_checkpoint(
        os.path.join(model_path, 'model.ckpt-best'))
    config_file = [x for x in os.listdir(model_path) if x.endswith('.json')][0]
    config = json.load(open(os.path.join(model_path, config_file)))

    loaded_params = {k: config[k] for k in params.keys()}

    tfbert = BertModel(shared_layer=True, **loaded_params)

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
    parser.add_argument('--input',
                        required=True,
                        type=str,
                        help='input tf1 bert dir')
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='output tf2 bert dir')
    args = parser.parse_args()

    model_path = args.input
    bert = BERT(model_path)
    bert._set_inputs(tf.keras.backend.placeholder((None, None),
                                                  dtype='string'))
    bert.save(args.output, include_optimizer=False)


main()
