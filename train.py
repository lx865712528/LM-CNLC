# -*- coding: utf-8 -*-

import _pickle as cPickle
import os
import pprint
import time

import numpy as np
import tensorflow as tf

from models.charrnn import CharRNN
from utils import TextLoader

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 20, "Epoch to train [20]")
flags.DEFINE_integer("num_units", 300, "The dimension of char embedding matrix [300]")
flags.DEFINE_integer("batch_size", 64, "The size of batch [64]")
flags.DEFINE_integer("rnn_size", 512, "RNN size [512]")
flags.DEFINE_integer("layer_depth", 2, "Number of layers for RNN [2]")
flags.DEFINE_integer("seq_length", 25, "The # of timesteps to unroll for [25]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate [1e-3]")
flags.DEFINE_string("rnn_type", "GRU", "RNN type [RWA, RAN, LSTM, GRU]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate [0.5]")
flags.DEFINE_float("grad_clip", 5.0, "Grad clip [5.0]")
flags.DEFINE_float("early_stopping", 2, "early stop after the perplexity has been "
                                        "detoriating after this many steps. If 0 (the "
                                        "default), do not stop early.")
flags.DEFINE_string("dataset_name", "normal", "The name of datasets [normal]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("export", False, "Export embedding")
FLAGS = flags.FLAGS


def run_epochs(sess, x, y, model, is_training=True):
    start = time.time()
    feed = {model.input_data: x, model.targets: y, model.is_training: is_training}

    if is_training:
        extra_op = model.train_op
    else:
        extra_op = tf.no_op()

    fetchs = {"loss": model.loss,
              "extra_op": extra_op}

    res = sess.run(fetchs, feed)
    end = time.time()

    return res, end - start


def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        print(" [*] Creating checkpoint directory...")
        os.makedirs(FLAGS.checkpoint_dir)

    data_loader = TextLoader(os.path.join(FLAGS.data_dir, FLAGS.dataset_name),
                             FLAGS.batch_size, FLAGS.seq_length)
    vocab_size = data_loader.vocab_size + 1

    with tf.variable_scope('model'):
        train_model = CharRNN(vocab_size, FLAGS.batch_size, FLAGS.rnn_size,
                              FLAGS.layer_depth, FLAGS.num_units, FLAGS.rnn_type,
                              FLAGS.seq_length, FLAGS.keep_prob,
                              FLAGS.grad_clip)

    with tf.variable_scope('model', reuse=True):
        valid_model = CharRNN(vocab_size, FLAGS.batch_size, FLAGS.rnn_size,
                              FLAGS.layer_depth, FLAGS.num_units, FLAGS.rnn_type,
                              FLAGS.seq_length, FLAGS.keep_prob,
                              FLAGS.grad_clip)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        train_model.load(sess, FLAGS.checkpoint_dir, FLAGS.dataset_name)

        best_val_pp = float('inf')
        best_val_epoch = 0
        valid_loss = 0
        valid_perplexity = 0
        start = time.time()

        if FLAGS.export:
            print("Eval...")
            final_embeddings = train_model.embedding.eval(sess)
            emb_file = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, 'emb.npy')
            print("Embedding shape: {}".format(final_embeddings.shape))
            np.save(emb_file, final_embeddings)

        else:
            if not os.path.exists(FLAGS.log_dir):
                os.makedirs(FLAGS.log_dir)
            with open(FLAGS.log_dir + "/" + FLAGS.dataset_name + "_hyperparams.pkl", 'wb') as f:
                cPickle.dump(FLAGS.__flags, f)
            for e in range(FLAGS.num_epochs):
                data_loader.reset_batch_pointer()
                sess.run(tf.assign(train_model.lr, FLAGS.learning_rate))
                for b in range(data_loader.num_batches):
                    x, y = data_loader.next_batch()
                    res, time_batch = run_epochs(sess, x, y, train_model)
                    train_loss = res["loss"]
                    train_perplexity = np.exp(train_loss)
                    iterate = e * data_loader.num_batches + b
                    print(
                        "{}/{} (epoch {}) loss = {:.2f}({:.2f}) perplexity(train/valid) = {:.2f}({:.2f}) time/batch = {:.2f} chars/sec = {:.2f}k" \
                            .format(e * data_loader.num_batches + b,
                                    FLAGS.num_epochs * data_loader.num_batches,
                                    e, train_loss, valid_loss, train_perplexity, valid_perplexity,
                                    time_batch, (FLAGS.batch_size * FLAGS.seq_length) / time_batch / 1000))
                valid_loss = 0
                for vb in range(data_loader.num_valid_batches):
                    res, valid_time_batch = run_epochs(sess, data_loader.x_valid[vb], data_loader.y_valid[vb],
                                                       valid_model, False)
                    valid_loss += res["loss"]
                valid_loss = valid_loss / data_loader.num_valid_batches
                valid_perplexity = np.exp(valid_loss)
                print("### valid_perplexity = {:.2f}, time/batch = {:.2f}".format(valid_perplexity, valid_time_batch))
                if valid_perplexity < best_val_pp:
                    best_val_pp = valid_perplexity
                    best_val_epoch = iterate
                    train_model.save(sess, FLAGS.checkpoint_dir, FLAGS.dataset_name)
                    print("model saved to {}".format(FLAGS.checkpoint_dir))
                if iterate - best_val_epoch > FLAGS.early_stopping:
                    print('Total time: {}'.format(time.time() - start))
                    break


if __name__ == '__main__':
    tf.app.run()
