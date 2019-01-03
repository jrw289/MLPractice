# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:36:14 2019

@author: Jake Welch
"""

import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from numpy import zeros

#Load the diabetes data set as a baseline 
#Data has 10 input variables 
raw_data = datasets.load_diabetes().data
targets = datasets.load_diabetes().target

#Create a linear regression model to take a look at the variables 
lin_reg = linear_model.LinearRegression()

r2_scores = zeros((10,2))

#For each of the variables, see what the score of a linear regression is
#I am using the train_test_split to randomize the training sets, but since
#    that is splitting the data separately for each variable, how valid is it?
for a in range(10):
    #Pull the data from column a
    temp_set = raw_data[:,a] 
    temp_train_x, temp_test_x, temp_train_y, temp_test_y = train_test_split(temp_set,targets,test_size = 0.1)
    
    #Need to reshape these arrays for linear regression algorithm to accept them 
    temp_train_x = temp_train_x.reshape(-1,1)
    temp_train_y = temp_train_y.reshape(-1,1)
    
    lin_reg.fit(temp_train_x,temp_train_y)
    
    #Predict output on the test data set
    temp_test_x = temp_test_x.reshape(-1,1)
    temp_test_y = temp_test_y.reshape(-1,1)
    
    temp_pred_y = lin_reg.predict(temp_test_x)
    
    #Measure the scores for each variable
    temp_r2 = r2_score(temp_test_y,temp_pred_y)
    r2_scores[a,:] = a, temp_r2
    print('For variable {}, R^2 score is {}\n'.format(a+1,temp_r2))
    
#Sort the columns by the best scores 
temp_ind = r2_scores[:,1].argsort()
ranked_r2 = r2_scores[temp_ind[::-1]]