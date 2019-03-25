# !usr/bin/env python
# -*- coding: utf-8 -*-
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
'''
以下为数据预处理，将index变为日期时间
'''
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')
def data_preprocessing():
    dataset = read_csv('raw.csv', parse_dates=[['year', 'month', 'day', 'hour']], infer_datetime_format=True,date_parser=parse)
    dataset.drop('No',axis=1, inplace=True)
    dataset.columns  = ['date','pollution', 'dew', 'temp', 'press', 'wnd_dir','wnd_spd', 'snow', 'rain']
    dataset.set_index(['date'], inplace=True)
    # dataset.index.name = 'date'
    dataset['pollution'].fillna(0, inplace=True)
    dataset = dataset[24:]
    print(dataset.head(5))
    dataset.to_csv('pollution.csv')

def show_data():
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    groups = [0, 1, 2, 3, 5, 6, 7]
    i =1
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups),1,i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

def series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]#列的数量
    df = DataFrame(data)
    cols, names = list(), list()
    #input sequence(t-n, ...,t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)'%(j+1,i))for j in range(n_vars)]
    #forecast sequence(t,...,t+1)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def load_dataset():
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    #integer encode direction,
    encoder = LabelEncoder()#对不连续的数字或者文本进行编号，eg:(1,1,100,25)-->(0,0,3,2)
    values[:, 4] = encoder.fit_transform(values[:, 4])
    #ensure all values is float
    values = values.astype('float32')
    #normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    #frame as supervised
    reframed = series_to_supervised(scaled, 1, 1)
    #drop columns we don't wanna predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    # print(reframed.head())
    return reframed, scaler

def split_data(n_train_hours = 365 * 24):
    #split into train and test sets
    reframed, scaler = load_dataset()
    values = reframed.values
    train = values[:n_train_hours,:]
    test = values[n_train_hours:,:]
    #split into input and output
    train_X, train_y = train[:,:-1],train[:,-1]
    test_X, test_y = test[:, :-1],test[:, -1]
    #reshape input to be 3D[samples, timesteps, features],
    # time_steps: 是输入时间序列的长度，即用多少个连续样本预测一个输出。
    # 如果你希望用连续m个序列（每个序列即是一个原始样本），那么就应该设为m。
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)
    return train_X, train_y, test_X, test_y, scaler

def LstmModel(epoch=50, batch_size=72):
    train_X, train_y, test_X, test_y, scaler = split_data()
    model = Sequential()
    #定义50个具有50个神经元的LSTM
    model.add(LSTM(50, input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X,train_y,epochs=epoch,batch_size=batch_size, validation_data=(test_X,test_y))
    #plot histroy
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model, test_X, test_y, scaler

def evaluate_model():
    model, test_X, test_y, scaler = LstmModel()
    #make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))
    #invert scaling for forecast
    inv_yhat = concatenate((yhat,test_X[:,1:]),axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    #invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    #calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(rmse)


if __name__ == '__main__':
    evaluate_model()



