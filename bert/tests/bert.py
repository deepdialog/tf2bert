import tensorflow as tf
from bert.bert import Bert
from .config import params


def test():

    bert = Bert(**params)

    batch_size = 4
    for max_length in (1, 100, 512, 768):
        x = tf.random.uniform((batch_size, max_length),
                              minval=0,
                              maxval=100,
                              dtype=tf.int32)
        t = tf.random.uniform((batch_size, max_length),
                              minval=0,
                              maxval=2,
                              dtype=tf.int32)

        # import pdb; pdb.set_trace()
        print(bert([x, t], pooler=True).shape)
    # print(len(bert.weights))
    # for x in bert.weights:
    #     print(x.name)


if __name__ == "__main__":
    test()
