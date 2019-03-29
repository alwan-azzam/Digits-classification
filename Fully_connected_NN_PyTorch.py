#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AzzamAlwan
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

'''*************************************************************************'''
class Net(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return F.log_softmax(x)
'''*************************************************************************'''
'''*************************************************************************'''
# main
dig = load_digits()
x_train, x_val, y_train, y_val = train_test_split(dig.data, dig.target, test_size=0.1, random_state=10) 

input_size = x_train.shape[1]
hidden_size = 128
output_size = np.max(y_train)+1
learning_rate = 0.1
size_batch = 32
epochs = 500
maxi = np.max(x_train)

x_train = np.array(x_train/maxi, dtype=np.float32) # data normalisation
x_train = torch.from_numpy(x_train) # to torch

x_val = np.array(x_val/maxi, dtype=np.float32) # data normalisation
x_val = torch.from_numpy(x_val) # to torch

y_train = torch.from_numpy(np.array(y_train))# to torch
y_val = torch.from_numpy(np.array(y_val)) #to torch

model = Net(input_size, hidden_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
criterion = nn.NLLLoss() # Cross-entropy
'''*************************************************************************'''
for iter in range(epochs):
    print(iter)
    for i in np.arange(0, len(x_train)-size_batch, size_batch):
        
        data = x_train[i : i+size_batch, : ]
        target = y_train[i : i+size_batch]
        
        optimizer.zero_grad()
        y_pred = model(data)      
        loss = criterion(y_pred, target)
        print(i, loss.item())      
        loss.backward()
        optimizer.step()


correct = 0
model_out = model(x_val)
y_pred = model_out.data.max(1)[1] 
correct += y_pred.eq(y_val).sum().item()
print("Test accuracy : ", correct/len(y_val))
