# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf#pip install --upgrade tensorflow==1.2.1升级tensorflow指定版本
from tensorflow.models.rnn.ptb import reader
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
DATA_PATH = "path/to/ptb/data"
HIDDEN_SIZE = 200 #隐藏层的规模
NUM_LAYERS = 2 #DRNN中LSTM结构的层数
VOCAB_SIZE = 10000 #词典规模，加上语句结束符和稀有单词结束符总共10000
LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20  #训练数据BATCH大小
TRAIN_NUM_STEPS = 35    #训练数据截断长度
#在测试的时候不需要使用截断
EVAL_BATCH_SIZE = EVAL_NUM_STEP = 1
NUM_EPOCH = 2 #使用训练数据的轮数
KEEP_DROP =0.5 #节点不被dropout的概率
MAX_GRAD_NORM =5 #用于控制梯度膨胀的参数


#定义一个PTBMODEL类来描述模型，方便维护循环神经网络中的状态
class PTBMODEL:
    def __init__(self,batch_size,num_steps,is_training = True):
        self.batch_size = batch_size
        self.num_steps = num_steps
        #定义输入层，维度为batch_size* num_steps
        self.input_data = tf.placeholder(tf.int32,shape=[batch_size,num_steps])
        #定义预期输出。它的维度和ptb_iterrattor输出的正确答案维度是一样的。
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])
        #定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=KEEP_DROP)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(NUM_LAYERS)])
        #初始化初始状态
        self.initial_state = cell.zero_state(batch_size,tf.float32)
        #将单词ID转换为单词向量，总共有VOCAB_SIZE个单词，每个单词向量的维度为HIDDEN_SIZE，所以embedding参数的维度为
        #VOCAB_SIZE*HIDDEN_SIZE
        embedding = tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])
        #将原本batch_size * num_steps个单词ID转化为单词向量，转化后的输入层维度为batch_size * num_steps * HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)
        #只在训练时使用dropout
        if is_training:
            inputs  = tf.nn.dropout(inputs,KEEP_DROP)
        #定义输出列表，在这里现将不同时刻LSTM结构的输出收集起来，再通过一个全连接层得到最终输出
        output = []
        #state 存储不同batch中LSTM的状态，并且初始化为0.
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step  in range(num_steps):
                if time_step > 0 :
                    tf.get_variable_scope().reuse_variables()
                cell_output,state = cell(inputs[:,time_step,:],state)
                #将当前输出加入输出队列
                output.append(cell_output)
        #把输出队列展开成[batch,hidden_size*num_steps]的形状，然后再reshape成【batch*num_steps,hidden_size】的形状。
        output = tf.reshape(tf.concat(output,1),[-1,HIDDEN_SIZE])
        #将从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每一时刻上都是一个长度为VOCAB_SIZE的数组
        #经过SoftMax层之后表示下一个位置是不同单词的概率。
        weight = tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
        baias  =  tf.get_variable("bias",[VOCAB_SIZE])
        logits = tf.matmul(output,weight) + baias
        #定义交叉熵损失函数

        loss  = sequence_loss_by_example([logits],[tf.reshape(self.targets,[-1])],
                                                                   [tf.ones([batch_size*num_steps],dtype=tf.float32)])
        '''
                 给损失函数加上L2范数防止过拟合
                 '''
        # L2 regularization for weights and biases
        # reg_loss = 0
        # for tf_var in tf.trainable_variables():
        #     if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
        #         reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        # loss += lambda_l2_reg * reg_loss

        #计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss)/batch_size
        self.final_state = state
        #只在训练模型是定义反向传播操作
        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        #通过clip_by_global_norm函数控制梯度的大小，避免梯度膨胀的问题
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost,trainable_variables),MAX_GRAD_NORM)
        #定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

#使用给定的模型model在数据data上运行train_op并返回全部数据上的perplexity值

def run_epoch(session,model,data,train_op,output_log):
    #计算perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    #使用当前数据训练或者测试模型
    for step ,(x,y) in  enumerate(reader.ptb_iterator( data,model.batch_size,model.num_steps)):
        cost,state,_ = session.run([model.cost,model.final_output,model.train_op],{
            model.input_data:x,model.targets:y,
            model.initial_state:state
        })
        total_costs += cost
        iters += model.num_steps
        #只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %s steps ,perplexity is %.3f"%(step,np.exp(total_costs/iters)))

    #返回给定模型在给定数据上的perplexity
    return np.exp(total_costs/iters)


def main(_):
    #获取原始数据
    train_data,valid_data,test_data = reader.ptb_raw_data(DATA_PATH)
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    #定义训练用的循环神经网络模型
    with tf.variable_scope("language_model",reuse=True,initializer=initializer):
        train_model = PTBMODEL(TRAIN_BATCH_SIZE,TRAIN_NUM_STEPS,is_training=True)
    #定义评估用的循环神经网络模型
    with tf.variable_scope("language_model",reuse=True,initializer=initializer):
        eval_model = PTBMODEL(EVAL_BATCH_SIZE,EVAL_NUM_STEP,is_training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print("In iteration:%s"%(i+1))
            #在所有训练数据上训练RNN
            run_epoch(sess,train_model,train_data,train_model.train_op,True)
            #使用验证集评测模型效果
            valid_perplexity = run_epoch(sess,eval_model,valid_data,tf.no_op(),False)
            print("Epoch %s ,Validation perplexity :%.3f"%(i+1,valid_perplexity))
        # 最后使用测试集验证模型效果
        test_perplexity = run_epoch(sess,eval_model,valid_data,tf.no_op(),False)
        print("TEST perplexity :%.3f"%(test_perplexity))

if __name__ == '__main__':
    tf.app.run()