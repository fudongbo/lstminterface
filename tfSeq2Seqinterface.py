#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import os
import copy
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

lt = ['cnt', 'income_after_dis']
savePath = 'Models/'
path = 'D:\\Predictionofsales\\' + '20190104114046.csv'
df = pd.read_csv(path, encoding='utf-8', engine='python')
df.dropna(how='any')
# reg_name_list = df['reg_name'].unique().tolist()
# reg_name_list = ['东北','华北','华东','华南','华中','西北','西南']
reg_name_list = ['dongbei','huabei','huadong','huanan','huazhong','xibei','xinan']
grouped = df.groupby(['reg_name'])
value = dict(list(grouped))
top = 123 #取前top个数据
cut_point = 0
howfar = 91
# print(value['东北']['cnt'][:5])
input_seq_len = 15
output_seq_len = 20
training = False#是否训练，true需要训练，false使用之前训练好的模型
isplot_orgin = True
def data_processing(raw_data, scale=True):
    if scale == True:
        return (raw_data - np.mean(raw_data)) / np.std(raw_data)  # 标准化
    else:
        return (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))  # 极差规格化

def scale_inv(data,raw_data, scale=True):
    if scale == True:
        return raw_data * np.std(data) + np.mean(data)
    else:
        return raw_data * (np.max(data) - np.min(data)) + np.min(data)
orgin_data = np.array( list(value['dongbei']['cnt'][:]))
y = np.array(data_processing(orgin_data))
x = np.array(range(len(orgin_data)))
train_data_x = x[:top]
if isplot_orgin == True:
    l1, = plt.plot(x[:top], orgin_data[:top], 'y', label = 'training samples')
    l2, = plt.plot(x[top:], orgin_data[top:], 'c--', label = 'test samples')
    plt.legend(handles = [l1, l2], loc = 'upper left')
    plt.show()
def true_signal(x):
    return np.array(orgin_data[x])
def generate_y_values(x):
    return np.array(y[x])
def generate_train_samples(x = train_data_x, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_start_points = len(x) - input_seq_len - output_seq_len#训练的起点最大的可选择范围
    start_x_idx = np.random.choice(range(total_start_points), batch_size)#在可选范围中随机找batch_size个数，作为训练起点

    input_seq_x = [x[i:(i+input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]

    input_seq_y = [generate_y_values(x) for x in input_seq_x]
    output_seq_y = [generate_y_values(x) for x in output_seq_x]

    #batch_x = np.array([[true_signal()]])
    return np.array(input_seq_y), np.array(output_seq_y)

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003#L2范数的系数

## Network Parameters
# length of input signals
input_seq_len = 15
# length of output signals
output_seq_len = 20
# size of LSTM Cell
hidden_dim = 64
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

def build_graph(feed_previous = False):

    tf.reset_default_graph()

    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),

    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))#输入为词相量时，input_dim则理解为词相量长度
               for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):

          with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=dtypes.float32,
                              scope=None):

          with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():#tf.trainable_variables()是返回所有需要训练的变量的列表，tf.all_variables()是返回所有变量列表
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        train_op = optimizer,
        loss=loss,
        saver = saver,
        reshaped_outputs = reshaped_outputs,
        )

total_iteractions = 100
batch_size = 16
KEEP_RATE = 0.5
train_losses = []
val_losses = []

rnn_model = build_graph(feed_previous=False)

# saver = tf.train.Saver()
if training == True:
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(total_iteractions):
            batch_input, batch_output = generate_train_samples(batch_size=batch_size)
            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            print(loss_t)

        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess, os.path.join(savePath, 'univariate_ts_model0'))

    print("Checkpoint saved at: ", save_path)

test_seq_input = true_signal(train_data_x[-input_seq_len:])

rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    rnn_model['saver']().restore(sess, os.path.join(savePath, 'univariate_ts_model0'))
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    final_preds = np.concatenate(final_preds, axis = 1)
l1, = plt.plot(range(top), true_signal(train_data_x[:top]), label = 'Training truth')
l2, = plt.plot(range(top, top+output_seq_len), orgin_data[top:top+output_seq_len], 'yo', label = 'Test truth')
l3, = plt.plot(range(top, top+output_seq_len), scale_inv(orgin_data, final_preds.reshape(-1)), 'ro', label = 'Test predictions')
plt.legend(handles=[l1, l2, l3], loc='lower left')
plt.show()
print('final_preds:  ', scale_inv(orgin_data, final_preds.reshape(-1)))
print('real:  ', orgin_data[top:top+output_seq_len])
