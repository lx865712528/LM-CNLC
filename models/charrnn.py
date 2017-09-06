import inspect

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from models.base import Model
from models.ran_cell import RANCell
from models.rwa_cell import RWACell


class SwitchableDropoutWrapper(rnn.DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0, seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell,
                                                       input_keep_prob=input_keep_prob,
                                                       output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                          for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state


class CharRNN(Model):
    def __init__(self, vocab_size=1000, batch_size=100,
                 rnn_size=1024, layer_depth=2, num_units=100,
                 rnn_type="GRU", seq_length=50, keep_prob=0.9,
                 grad_clip=5.0):

        Model.__init__(self)

        # RNN
        self._layer_depth = layer_depth
        self._keep_prob = keep_prob
        self._batch_size = batch_size
        self._num_units = num_units
        self._seq_length = seq_length
        self._rnn_size = rnn_size
        self._vocab_size = vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="inputs")
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")
        self.is_training = tf.placeholder('bool', None, name="is_training")

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size],
                                        initializer=tf.truncated_normal_initializer(stddev=1e-4))
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

            def create_cell(device):
                if rnn_type == "GRU":
                    cell = rnn.GRUCell(rnn_size)
                elif rnn_type == "LSTM":
                    if 'reuse' in inspect.signature(tf.contrib.rnn.BasicLSTMCell.__init__).parameters:
                        cell = rnn.LayerNormBasicLSTMCell(rnn_size, forget_bias=0.0,
                                                          reuse=tf.get_variable_scope().reuse)
                    else:
                        cell = rnn.LayerNormBasicLSTMCell(rnn_size, forget_bias=0.0)
                elif rnn_type == "RWA":
                    cell = RWACell(rnn_size)
                elif rnn_type == "RAN":
                    cell = RANCell(rnn_size, normalize=self.is_training)
                cell = SwitchableDropoutWrapper(rnn.DeviceWrapper(cell, device="/gpu:{}".format(device)),
                                                is_train=self.is_training)
                return cell

            self.cell = cell = rnn.MultiRNNCell([create_cell(i) for i in range(layer_depth)], state_is_tuple=True)

            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("embedding", [vocab_size, num_units])
                inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
                inputs = tf.contrib.layers.dropout(inputs, keep_prob, is_training=self.is_training)

        with tf.variable_scope("output"):
            self.initial_state = cell.zero_state(batch_size, tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    time_major=False,
                                                    swap_memory=True,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32)
            output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

        with tf.variable_scope("loss"):
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

            flat_targets = tf.reshape(tf.concat(self.targets, 1), [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=flat_targets)

            self.loss = tf.reduce_mean(loss)
            self.final_state = last_state
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def go_step(self, sess, state, little_step):
        '''
        State change
        :param sess: tf session
        :param state: model state
        :param little_step: int for next char
        :return: 1d probability tensor, loss and destination state
        '''
        little_step_data = np.zeros(shape=(1, 1))
        little_step_data[0][0] = little_step
        # assert type(little_step) == np.ndarray and little_step.shape == (1, 1)
        feed = {self.input_data: little_step_data, self.initial_state: state, self.is_training: False}
        fetchs = {"probs": self.probs, "loss": self.loss, "state": self.final_state}
        res = sess.run(fetchs, feed)
        res["probs"] = res["probs"][0]
        return res


if __name__ == "__main__":
    model = CharRNN()
