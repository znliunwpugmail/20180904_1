import tensorflow as tf
import numpy as np
import os
import random
from collections import deque
import time

EPISDOE = 10000
STEP = 10000
ENV_NAME = 'MountainCar-v0'
BATCH_SIZE = 2
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.1
REPLAY_SIZE = 50000
TRAIN_START_SIZE = 50
GAMMA = 0.9
def get_weights(shape):
    weights = tf.truncated_normal( shape = shape, stddev = 0.01 )
    return tf.Variable(weights)

def get_bias(shape):
    bias = tf.constant( 0.01, shape = shape )
    return tf.Variable(bias)

class DQN():
    def __init__(self,action_dim,state_dim,save_path = 'log/train'):
        self.epsilon_step = ( INIT_EPSILON - FINAL_EPSILON ) / 10000
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.neuron_num = 10
        self.replay_buffer = deque()
        self.epsilon = INIT_EPSILON
        self.sess = tf.InteractiveSession()
        self.init_network()
        self.merged = tf.summary.merge_all()
        self.save_path = None
        self.sess.run( tf.global_variables_initializer() )
        self.saver = tf.train.Saver()
        # self.merged = None
        if save_path is not None:
            self.save_path = save_path
            if os.path.exists(self.save_path):
                self.save_path = self.save_path+'_1'
        self.train_writer = tf.summary.FileWriter(self.save_path, self.sess.graph)

    def init_network(self):#神经网络构建，最后输出的value是三个action的概率，输出的cost是输出值和reward的平方差函数
        self.input_layer = tf.placeholder( tf.float32, [ None, self.state_dim ] )
        self.action_input = tf.placeholder( tf.float32, [None, self.action_dim] )
        self.y_input = tf.placeholder( tf.float32, [None] )
        with tf.name_scope('summary'):
            w1 = get_weights( [self.state_dim, self.neuron_num] )
            b1 = get_bias([self.neuron_num])
            hidden_layer = tf.nn.relu( tf.matmul( self.input_layer, w1 ) + b1 )
            w2 = get_weights( [ self.neuron_num, self.action_dim ] )
            b2 = get_bias( [ self.action_dim ] )
            self.Q_value = tf.matmul( hidden_layer, w2 ) + b2#最终输出每个action的概率
            value = tf.reduce_sum( tf.multiply( self.Q_value, self.action_input ), reduction_indices = 1 )
            self.cost = tf.reduce_mean( tf.square( value - self.y_input ) )
            self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)
        return

    def percieve(self, state, action, reward, next_state, is_save = False,step = None):#将action转为热一编码
        one_hot_action = np.zeros( [ self.action_dim ] )
        one_hot_action[action] = 1
        self.replay_buffer = [ state, one_hot_action, reward, next_state]
        # self.replay_buffer.append( [ state, one_hot_action, reward, next_state, done ] )
        self.train(is_save,step)

    def train(self,is_save = False,step = None):
        state_batch = self.replay_buffer[0]
        action_batch = self.replay_buffer[1]
        reward_batch = self.replay_buffer[2]
        next_state_batch = self.replay_buffer[3]

        y_batch = []
        next_state_reward = self.Q_value.eval( feed_dict = { self.input_layer : [next_state_batch]} )
        y_batch.append( reward_batch + GAMMA * np.max( next_state_reward) )

        y_batch = np.array(y_batch)
        now_time = time.time()
        self.optimizer.run(
            feed_dict = {
                self.input_layer:[state_batch],
                self.action_input:[action_batch],
                self.y_input:y_batch
            }
        )
        end_time = time.time()
        print('run the network',end_time-now_time)
        # print(self.merged)
        if is_save == True and step is not None:
            self.sess.run(self.cost,
                feed_dict={
                    self.input_layer: [state_batch],
                    self.action_input: [action_batch],
                    self.y_input: y_batch
                }
            )
            # self.train_writer.add_summary(summary,step)
            self.saver.save(self.sess,save_path=self.save_path+'model.ckpt')

        return
    def get_greedy_action(self, state):
        value = self.Q_value.eval( feed_dict = { self.input_layer : state } )
        return np.argmax( value,axis=1 )

    def get_action(self, state):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step
        if random.random() < self.epsilon:
            return np.array([random.randint(0, self.action_dim-1) for i in range(len(state))])
        else:
            return self.get_greedy_action(state)
