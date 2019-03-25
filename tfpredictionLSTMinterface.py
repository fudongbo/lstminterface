# -*- coding:utf-8 -*-
#!/usr/bin/env python
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
def delemodels(rootdir):
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
class predictLstm:
    def __init__(self, data, HIDDEN_SIZE = 30, NUM_LAYERS = 2,TIMESTEPS = 10,
                 TRAINING_STEPS = 3000, BATCH_SIZE = 32, learning_rate = 0.1, scale=True, use_peepholes=False):
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NUM_LAYERS = NUM_LAYERS
        self.TIMESTEPS = TIMESTEPS
        self.TRAINING_STEPS = TRAINING_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.learning_rate = learning_rate
        self.data = data
        self.dataprocess = []
        self.scale=scale
        self.use_peepholes = use_peepholes

    def data_processing(self, raw_data):
        if self.scale == True:
            return (raw_data - np.mean(raw_data)) / np.std(raw_data)  # 标准化
        else:
            return (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))  # 极差规格化

    def generate_data(self, seq):
        X = []  # 初始化输入序列X
        Y = []  # 初始化输出序列Y
        for i in range(len(seq) - self.TIMESTEPS - 1):
            X.append([seq[i:i + self.TIMESTEPS]])  # 从输入序列第一期出发，等步长连续不间断采样
            Y.append([seq[i + self.TIMESTEPS]])  # 对应每个X序列的滞后一期序列值
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def LstmCell(self):
        if self.use_peepholes == False:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE, state_is_tuple=True)  #
        else:
            lstm_cell = tf.contrib.rnn.LSTMCell(self.HIDDEN_SIZE, use_peepholes=True, state_is_tuple=True) # use peep-hole model
        return lstm_cell

    def lstm_model(self, X, y):
        '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，我们这里只有1层'''
        cell = tf.contrib.rnn.MultiRNNCell([self.LstmCell() for _ in range(self.NUM_LAYERS)])
        output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        output = tf.reshape(output, [-1, self.HIDDEN_SIZE])
        predictions = tf.contrib.layers.fully_connected(output, 1, None)
        labels = tf.reshape(y, [-1])
        predictions = tf.reshape(predictions, [-1])
        loss = tf.losses.mean_squared_error(predictions, labels)
        # L2 regularization for weights and biases
        lambda_l2_reg = 0.1
        ##方法一：手动自己算
        # reg_loss = 0
        # for tf_var in tf.trainable_variables():  # tf.trainable_variables()是返回所有需要训练的变量的列表，tf.all_variables()是返回所有变量列表
        #     if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
        #         reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        ##方法二：使用TensorFlow自带的函数
        # reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),
        #                                                   tf.trainable_variables())
        # loss += lambda_l2_reg * reg_loss
        train_op = tf.contrib.layers.optimize_loss(
            loss,
            # tf.contrib.framework.get_global_step(),
            tf.train.get_global_step(),
            optimizer='Adagrad',
            learning_rate=self.learning_rate)
        '''返回预测值、损失函数及优化器'''
        return predictions, loss, train_op

    def cutData(self, cut_point):
        data = self.data_processing(self.data)
        self.dataprocess = data
        if cut_point == 0:
            training_data = data
            test_data = data
        else:
            training_data = data[:int(len(data) * cut_point)]
            test_data = data[int(len(data) * cut_point)-self.TIMESTEPS:]
        return np.array(training_data, dtype=np.float32), np.array(test_data, dtype=np.float32)

    def scale_inv(self, raw_data):
        if self.scale == True:
            return raw_data * np.std(self.data) + np.mean(self.data)
        else:
            return raw_data * (np.max(self.data) - np.min(self.data)) + np.min(self.data)

    def prediction(self, regressor, howfar):
        final_result = []
        result = self.dataprocess[len(self.dataprocess)-howfar:]
        for i in range(howfar):
            predicted = regressor.predict(result)
            del result[0]
            result.append(predicted)
            final_result.append(predicted)
        return final_result

    def running(self, save_path,cut_point, howfar):
        training_data, test_data = self.cutData(cut_point)
        learn = tf.contrib.learn
        train_X, train_y = self.generate_data(training_data)
        test_X, test_y = self.generate_data(test_data)
        regressor = SKCompat(learn.Estimator(model_fn=self.lstm_model, model_dir=save_path))#
        regressor.fit(train_X, train_y, batch_size=self.BATCH_SIZE, steps=self.TRAINING_STEPS)
        predicted = np.array([[pred] for pred in regressor.predict(test_X)])
        final_result = np.array([[]],dtype= np.float32)
        result = np.array([test_X[-1]], dtype=np.float32)
        for i in range(howfar):
            tmp_predicted = np.array([[pred] for pred in regressor.predict(result)])
            result = np.array([np.hstack((np.delete(result[0], 0, axis=1), tmp_predicted))],dtype=np.float32)
            final_result = np.hstack((final_result, tmp_predicted))
        return self.scale_inv(predicted.flatten()), self.scale_inv(test_y.flatten()), self.scale_inv(final_result[0])

def parameters(HIDDEN_SIZE, NUM_LAYERS, learning_rate, save_path,cut_point, howfar,TRAINING_STEPS=3000, BATCH_SIZE=32, TIMESTEPS=10):
    delemodels(save_path)
    rmse = 0
    best_HIDDEN_SIZE = 0
    best_NUM_LAYERS = 0
    best_learning_rate = 0
    count = 0
    for hidden_s in HIDDEN_SIZE:
        for num_l in NUM_LAYERS:
            for lr in learning_rate:
                a = predictLstm(hidden_s, num_l, TIMESTEPS, TRAINING_STEPS, BATCH_SIZE, lr)
                tmp_predicted, tmp_test_y, tmp_final_result = a.running(save_path + str(count) + '/', cut_point, howfar)
                tmp = sqrt(mean_squared_error(tmp_test_y, tmp_predicted))
                if tmp < rmse or rmse == 0:
                    rmse  = tmp
                    best_HIDDEN_SIZE = hidden_s
                    best_NUM_LAYERS = num_l
                    best_learning_rate = lr
                    predicted = tmp_predicted
                    test_y = tmp_test_y
                    final_result = tmp_final_result
                count = count + 1
                del a
    return rmse,best_HIDDEN_SIZE, best_NUM_LAYERS, best_learning_rate, predicted, test_y, final_result

def save_final_results(dictionary, filename):
    dataframe = pd.DataFrame(dictionary)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filename,index=False,sep=',')

def read_file(path, ii):
    data = pd.read_csv(path)
    data = data.iloc[:, ii].tolist()
    return [x for x in data if x == x]

def mape(predicted, orgindata):
    tmp = 0
    N = len(predicted)
    for i in range(N):
        tmp += abs(predicted[i]/orgindata[i] - 1)
    return tmp/N

def dif(predicted, orgindata):
    N = len(orgindata)
    tmp = []
    for i in range(N):
        tmp.append(abs(predicted[i] - orgindata[i]))
    return tmp

if __name__ == '__main__':
    path = 'D:\\Predictionofsales\\' + 'international-airline-passengers.csv'
    cut_point = 0
    howfar = 7
    save_path = 'Models/'
    delemodels(save_path)  # 清空模型
    # tmp = read_file(path, 2)
    # tmp = [647087,610052,737766,546387,509127,566162,560567,602888,757897,749101,647087,610052,737766,546387,509127,566162,560567,602888,757897,749101]
    tmp = [6,73,16,12,20,23,218,51,5,204,153,93,26,85,51,4,85,102,96,39,5,6,9]#, 285, 1813, 869, 746, 181, 1]

    a = predictLstm(tmp)
    predicted, test_y, final_result = a.running(save_path, cut_point, howfar)
    print('testresults: ',  predicted, test_y, final_result)
    # print('mape:', mape(final_result, read_file(path, 3)))
    # save_final_results({'predicted':pd.Series(predicted),'test_y':pd.Series(test_y),'final_result':pd.Series(final_result)}, 'final_result1.csv')
    # rmse, best_HIDDEN_SIZE, best_NUM_LAYERS, best_learning_rate, predicted, test_y, final_result = parameters([30,40],[2],[0.1],path,save_path,cut_point,howfar)
    # print(rmse)
