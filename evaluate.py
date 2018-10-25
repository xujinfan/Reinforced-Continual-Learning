# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:01:59 2018

@author: Jason
"""
import tensorflow as tf
import numpy as np
import pdb


class evaluate:
    def __init__(self,task_list,args):
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.task_list = task_list
        self.stamps={}
        
    def evaluate_action(self,var_list,actions,task_id):
        with tf.Graph().as_default() as g:
            self.sess = tf.Session(graph=g)
            self.stamps[task_id-1] = [_.shape for _ in var_list]
            self.task_id = task_id
            with tf.name_scope("model"):
                self.x = tf.placeholder(tf.float32,shape=[None,784]) 
                self.y = tf.placeholder(tf.float32,shape=[None,10])
                fc1 = tf.Variable(tf.concat([var_list[0],tf.truncated_normal((var_list[0].shape[0],actions[0]),stddev=0.01)],axis=1))
                b1 = tf.Variable(tf.concat([var_list[1],tf.constant(0.1,shape=(actions[0],))],axis=0))
                transform1 = tf.Variable(tf.constant(1.0, shape=(var_list[0].shape[1] + actions[0],)))
                mask_fc1 = np.concatenate([np.zeros_like(var_list[0]),np.ones((var_list[0].shape[0],actions[0]))],axis=1)
                mask_b1 = np.concatenate([np.zeros_like(var_list[1]),np.ones((actions[0]))],axis=0)
                mask_transform1 = np.ones((var_list[0].shape[1] + actions[0]))

                
                old_shape = var_list[2].shape
                value = tf.concat([var_list[2],tf.truncated_normal((actions[0],old_shape[1]),stddev=0.01)],axis=0)
                fc2 = tf.Variable(tf.concat([value,tf.truncated_normal((actions[0]+old_shape[0],actions[1]),stddev=0.01)],axis=1))
                b2 = tf.Variable(tf.concat([var_list[3],tf.constant(0.1,shape=(actions[1],))],axis=0))
                transform2 = tf.Variable(tf.constant(1.0, shape=(old_shape[1] + actions[1],)))
                mask_fc2 = np.concatenate([np.concatenate([np.zeros_like(var_list[2]),np.ones((actions[0],old_shape[1]))],axis=0),np.zeros((actions[0]+old_shape[0],actions[1]))],axis=1)
                mask_b2 = np.concatenate([np.zeros_like(var_list[3]),np.ones((actions[1],))],axis=0)

                mask_transform2 = np.ones((old_shape[1] + actions[1]))
                fc3 = tf.Variable(tf.truncated_normal((var_list[4].shape[0]+actions[1],var_list[4].shape[1])))
                mask_fc3 = np.ones_like(fc3)
                b3 = tf.Variable(tf.constant(0.1,shape=(var_list[4].shape[1],)))
                mask_b3 = np.ones_like(b3)
            total_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            total_theta2 = [fc1, b1, fc2, b2, fc3, b3]
            h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(self.x,fc1,b1,name="fc1"))
            h_fc1_t1 = tf.multiply(h_fc1, transform1)
            h_fc2 = tf.nn.relu(tf.nn.xw_plus_b(h_fc1_t1,fc2,b2,name="fc2"))
            h_fc2_t2 = tf.multiply(h_fc2, transform2)
            h_fc3 = tf.nn.xw_plus_b(h_fc2_t2,fc3,b3,name="fc3")
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits = h_fc3)) + 0.0001*(tf.nn.l2_loss(fc1) + tf.nn.l2_loss(fc2) + tf.nn.l2_loss(fc3))
            
            if self.optimizer=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer=="rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer=="sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise Exception("please choose one optimizer")
            total_mask = [mask_fc1,mask_b1,mask_transform1,mask_fc2,mask_b2,mask_transform2,mask_fc3,mask_b3]
            grads_and_vars = optimizer.compute_gradients(loss,var_list= total_theta)
            grads_and_vars2 = self.apply_prune_on_grads(grads_and_vars, total_mask)
            train_step = optimizer.apply_gradients(grads_and_vars2)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,axis=1),tf.argmax(h_fc3,axis=1)),tf.float32))
            self.sess.run(tf.global_variables_initializer())
            
            l = len(self.task_list[0][1])
            for epoch in range(self.epochs):
                flag=0
                for _ in range(l//self.batch_size):
                    batch_xs, batch_ys = (self.task_list[task_id][0][flag:flag+self.batch_size],
                                          self.task_list[task_id][1][flag:flag+self.batch_size])
                    flag+=self.batch_size
                    self.sess.run(train_step,feed_dict={self.x:batch_xs,self.y:batch_ys})
                accuracy_val = self.sess.run(accuracy, feed_dict={self.x:self.task_list[task_id][2],
                                                                  self.y:self.task_list[task_id][3]})
                accuracy_test = self.sess.run(accuracy,feed_dict={self.x:self.task_list[task_id][4],
                                                                  self.y:self.task_list[task_id][5]})
                if epoch%4==0 or epoch==self.epochs-1:
                    print("task:%s,test accuracy:%s"%(task_id,accuracy_test))
            self.var_list = self.sess.run(total_theta2)
            self.stamps[task_id]=[_.shape for _ in self.var_list]
            self.sess.close()
            return (accuracy_val,accuracy_test)

    def conv2d(self,x, W): 
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    def apply_prune_on_grads(self,grads_and_vars, total_mask):
        for i in range(0,len(total_mask),2):
            grads_and_vars[i] = (tf.multiply(grads_and_vars[i][0], total_mask[i]),grads_and_vars[i][1])
            grads_and_vars[i+1] = (tf.multiply(grads_and_vars[i+1][0], total_mask[i+1]),grads_and_vars[i+1][1])
        return grads_and_vars

