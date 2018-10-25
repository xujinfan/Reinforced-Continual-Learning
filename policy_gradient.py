# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:45:08 2018

@author: Jason
"""
import torch
import torch.nn as nn
from torch.nn import init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from ipdb import set_trace
import argparse
import numpy as np


class PolicyEstimator(nn.Module):
    '''
    policy function approximator
    '''
    def __init__(self,args):
        super(PolicyEstimator,self).__init__()
        self.input_size = args.state_space
        self.state_space = args.state_space
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.actions_num = args.actions_num
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.
                            num_layers, batch_first=True)
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                # if 'weight' in p:     # uncomment to initialize weights only
                init.uniform(self.lstm.__getattr__(p), -0.08, 0.08)   # intra LSTM weights/ bias initialized
        self.hidden2output = nn.Linear(self.hidden_size, self.state_space)
        init.uniform(self.hidden2output.weight.data, -0.08, 0.08)     # weights to output from U[-0.08, 0.08]  
        init.uniform(self.hidden2output.bias.data, -0.08, 0.08)       # bias to output from U[-0.08, 0.08]
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = (Variable(torch.zeros(self.num_layers, self.batch_size,  self.hidden_size)),  # hidden = 0
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))         # tried with randN also, didn't work
        return hidden

    def forward(self,x):
        self.policy_actions = []
        self.actions = []
        sum_prob=[]
        for i in range(self.actions_num):
            if i==0:
                outputs,hidden = self.lstm(x.view(1,1,-1))
            else:
                outputs,hidden = self.lstm(x.view(1,1,-1),hidden)
            classifier = self.hidden2output(outputs[:,-1,:].view(-1))
            preds = F.softmax(classifier,dim=0)
            temp = F.log_softmax(classifier,dim=0)
            x = preds.view(1,1,-1)
            self.policy_actions.append(preds)
            max_index = np.argmax(preds.data)
            self.actions.append(max_index)
            sum_prob.append(temp[max_index])

        return self.policy_actions
    

class ValueEstimator(nn.Module):
    '''
    value function approximator
    '''
    def __init__(self,args):
        super(ValueEstimator,self).__init__()
        self.input_size = args.state_space
        self.fc = nn.Linear(self.input_size,1)
    def forward(self,x):
        out = self.fc(x)
        return torch.squeeze(out)


class Controller(object):
    '''
    update PolicyEstimator and ValueEstimator
    '''
    def __init__(self,args):
        self.policy_estimator = PolicyEstimator(args)
        self.value_estimator = ValueEstimator(args)
        self.state = Variable(torch.randn(1,1,args.state_space))
        self.epison = 0.5
        self.steps = 0
        self.args = args
        self.actions_values = []
        self.set_value(0.8)
        self.opt_policy = torch.optim.Adam(self.policy_estimator.parameters(), 
                       lr = args.lr)
        self.opt_value = torch.optim.Adam(self.value_estimator.parameters(), 
                       lr = 0.005)
        self.loss_value_fc = torch.nn.MSELoss()
        
    def get_actions(self):
        if self.args.method=="random":
            self.actions = np.random.choice(range(self.args.state_space), self.args.actions_num)
            return self.actions
        policy_actions = self.policy_estimator(self.state)
        self.sum_prob = []
        self.actions = []
        for i in range(self.args.actions_num):
            temp = policy_actions[i]
            prob = np.round(temp.data.numpy(),5)/sum(np.round(temp.data.numpy(),5))
            index = np.random.choice(range(self.args.state_space),p=prob)
            self.actions.append(index)
            self.sum_prob.append(policy_actions[i][index])
        
        return self.actions

    def get_value(self):
        out = self.value_estimator(self.state)
        return out

    def set_value(self,value):
        self.actions_values.append(value)

    def average_value(self):
        return np.mean(self.actions_values)

    def max_value(self):
        return np.max(self.actions_values)

    def train_controller(self,reward):
        if self.args.method=="random":
            return
        if self.args.bendmark == 'max':
            state_value = self.max_value()
        elif self.args.bendmark == 'average':
            state_value = self.average_value()
        elif self.args.bendmark == 'critic':
            state_value = self.get_value()
        else:
            raise Exception("please define the bendmark")
            
        target = Variable(torch.FloatTensor([float(reward)]))
        value_loss = self.loss_value_fc(state_value,target)
        self.opt_value.zero_grad()
        value_loss.backward(retain_graph=True)
        self.opt_value.step()
        
        self.opt_policy.zero_grad()
        advantage = reward - state_value.detach().numpy() 
        self.set_value(reward)
        for i,value in enumerate(self.sum_prob):
            if i==0:
                sum_prob = torch.log(value)
            else:
                sum_prob += torch.log(value)
        loss = -sum_prob*advantage
        loss = loss.reshape(-1)[0]
        loss.backward()
        self.opt_policy.step()
        
