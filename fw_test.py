import math
import os
import pickle

import numpy as np
import tensorflow as tf

from models.charrnn import CharRNN
from utils import UNK_ID, GO, EOS


def main(_):
    with open("./data/normal_hyperparams.pkl", 'rb') as f:
        config = pickle.load(f)

    with open("./data/normal_vocab.pkl", 'rb') as f:
        vocab_i2c = pickle.load(f)
        vocab_size = len(vocab_i2c)
        vocab_c2i = dict(zip(vocab_i2c, range(vocab_size)))

    with tf.variable_scope('normal'):
        model = CharRNN(vocab_size=vocab_size,
                           batch_size=1,
                           rnn_size=config['rnn_size'],
                           layer_depth=config['layer_depth'],
                           num_units=config['num_units'],
                           seq_length=1,
                           keep_prob=config['keep_prob'],
                           grad_clip=config['grad_clip'],
                           rnn_type=config['rnn_type'])


    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("checkpoint/normal")
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        print("Done!")

        while True:
            sentence = input()
            chars = [GO] + list(sentence) + [EOS]
            fw_ints = [vocab_c2i.get(c, UNK_ID) for c in chars]
            print(fw_ints)

            loss, _ = model.get_loss(sess, fw_ints)
            print("ppl", np.exp(-loss))


if __name__ == '__main__':
    tf.app.run()
