#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AzzamAlwan
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
'''*************************************************************************'''
dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=10)

'''*************************************************************************'''
# activation function
def RELU(s):
    return np.maximum(0,s)
# derivative of the relu activation function
def RELU_derv(z):
    return np.where(z <= 0, 0, 1)

# sigmoid function
def sigmoid(s):
    return 1/(1 + np.exp(-s))

#derivative of the sigmoid function
def sigmoid_derv(s):
    return s * (1 - s)

# softmax fucntion, that make the sum of the all vector elements equal to 1
def softmax(s):
    exps = np.exp(s)
    return exps/np.sum(exps, axis=1, keepdims=True)

def error(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

# loss function of the NN
def cross_entropy(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss
'''*************************************************************************'''
'''*************************************************************************'''
class MyNN:
    def __init__(self, data, label, neurons):
        self.neurons = neurons
        self.lr = 0.1
        input_size = data.shape[1]
        output_size = label.shape[1]

        self.w1 = np.random.randn(input_size, self.neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(self.neurons, output_size)
        self.b2 = np.zeros((1, output_size))

    def feedforward(self, data):
        z1 = np.dot(data, self.w1) + self.b1
        self.a1 = RELU(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(z2)
                
    def backprop(self, data, label):
        loss = cross_entropy(self.a2, label)
        print('loss :', loss)
        a2_delta = error(self.a2, label)
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * RELU_derv(self.a1)  
        #update
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0, keepdims=True)
        self.w1 -= self.lr * np.dot(data.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
        
    def predict(self, data):
        self.feedforward(data)
        return self.a2.argmax()
'''*************************************************************************'''
'''*************************************************************************'''
def Accuracy(data, label, model):
    acc = 0
    for x,y in zip(data, label):
        s = model.predict(x)
        if s == np.argmax(y):
            acc +=1
    return acc/len(data)*100		
'''*************************************************************************'''
# main funtion  
x_train_Norm = x_train/np.max(x_train)
x_val_Norm = x_val/np.max(x_train)
neurones_nomber = 128
y_train = np.array(y_train)
size_batch = 32
epochs = 500

model = MyNN(x_train_Norm, y_train , neurones_nomber)

for iter in range(epochs):
    print(iter)
    for i in np.arange(0, len(x_train_Norm)-size_batch, size_batch):
        model.feedforward(x_train_Norm[i : i+size_batch, : ])
        model.backprop(x_train_Norm[i : i+size_batch, : ], y_train[i : i+size_batch,:])
		
print("Test accuracy : ", Accuracy(x_val_Norm, np.array(y_val), model))


