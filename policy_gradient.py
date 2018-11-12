# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:45:08 2018

@author: Jason
"""

import numpy as np
import tensorflow as tf


class PolicyEstimator:
    """
    Policy Function approximator.
    """

    def __init__(self, args, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.args = args
            self.input_size = args.state_space
            self.state_space = args.state_space
            self.hidden_size = args.hidden_size
            self.num_layers = args.num_layers
            self.actions_num = args.actions_num
            self.state = tf.placeholder(dtype=tf.float32, shape=(None,self.state_space), name="states")
            self.actions = tf.placeholder(dtype=tf.int32, shape=(self.actions_num, ), name="actions")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            self.hidden2output_w = tf.Variable(tf.truncated_normal(shape=(self.hidden_size, self.state_space), stddev=0.01))
            self.hidden2output_b = tf.Variable(tf.constant(0.1,shape=(self.state_space,)))
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) for _ in range(self.num_layers)], state_is_tuple=True)
            hidden_state = cell.zero_state(1, dtype=tf.float32)
            inputs = self.state
            self.outputs = []
            with tf.variable_scope("LSTM"):
                for time_step  in range(self.actions_num):
                    (cell_output, hidden_state) = cell(inputs, hidden_state)
                    inputs = tf.nn.softmax(tf.nn.xw_plus_b(cell_output, self.hidden2output_w, self.hidden2output_b))
                    self.outputs.append(inputs)
            for time_step in range(self.actions_num):
                if time_step == 0:
                    picked_action_prob = self.outputs[time_step][0,self.actions[time_step]]
                else:
                    picked_action_prob = picked_action_prob*self.outputs[time_step][0,self.actions[time_step]]
            #loss and training op
            self.loss =  -tf.log(picked_action_prob)*self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess if sess else tf.get_default_session()
        return sess.run(self.outputs, {self.state: state})

    def update(self, state, target, actions, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.actions: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    """
    Value Function approximator.
    """

    def __init__(self, args, learning_rate=0.005, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state_space = args.state_space
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, self.state_space), name="states")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            self.state = tf.reshape(self.state, shape=(1, self.state_space))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess if sess else tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class Controller:
    def __init__(self, args, scope="Controller"):
        self.args = args
        self.state = np.random.random(size=(1, args.state_space))
        self.policy_estimator = PolicyEstimator(args)
        self.value_estimator = ValueEstimator(args)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_controller(self, reward):
        baseline_value = self.value_estimator.predict(self.state, self.sess)
        advantage = reward - baseline_value
        self.value_estimator.update(self.state, reward, self.sess)
        self.policy_estimator.update(self.state, advantage, self.actions, self.sess)

    def get_actions(self):
        action_probs = self.policy_estimator.predict(self.state, self.sess)
        self.actions = []
        for i in range(self.args.actions_num):
            prob = action_probs[i]
            action = np.random.choice(np.arange(self.args.state_space),p=prob[0])
            self.actions.append(action)
        return self.actions

    def close_session(self):
        self.sess.close()
