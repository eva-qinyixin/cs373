#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
from __future__ import division
from classifier import BinaryClassifier
import numpy as np
import pandas as pd
import utils

class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        self.up=0
        self.learning_step=args.lr
        self.max_iteration=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.b=0
        
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        
        #train_data[0] features
        #train_data[1] label
        words_bin_form=utils.get_feature_vectors(train_data[0])
        self.w=[0.0]*(len(words_bin_form[0]))
        #self.b=0
        
        #correct_count=0
        time=0
        
        #for i in range len(self.w):
            
        
        
        #print("train_data")
        #print(len(train_data)) #2
        #print(train_data)
        
        #print("train_data[0]") 
        #print(len(train_data[0])) #699
        #print(train_data[0])
        
        #error_found=True
        
        while time<self.max_iteration:  #& error_found:
           # error_found=False
            for i in range(len(words_bin_form)):
                x=words_bin_form[i]
                sumx=sum([words_bin_form[i][j] for j in range(self.dim)] )
                y=train_data[1][i]
            #index=np.random.randint(0,(len(train_data[1])-1))
           # x=list(words_bin_form[index])
            #x.append(1.0)
            #y=train_data[1][index]
            
                wx_b= sum([self.w[j]*x[j]/sumx for j in range(len(self.w))])+self.b
                
            
                si=np.sign(wx_b)
            
            
                time+=1;
            
                if y==si:
                     break
                else:
                   # error_found=True
                    for j in range(len(self.w)):
                        self.w[j]+=self.learning_step*y*x[j]
                        self.b+=self.learning_step*y
                    
            #if wx*y>0:
               # correct_count+=1
                #if correct_count>self.max_iteration:
                  #  break
                #continue    
             
           # for i in range(len(self.w)):
                  #  self.w[i]+=self.learning_step*(y*x[i])
            #print("finishing training perceptron")
            #print("self.w")
            #print(self.w)
            #print("self.b")
            #print(self.b)
                    
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
#            print("inside perceptron predict")
            #print("test_x")
            #print(len(test_x))#301
            #print(len(test_x[0])) #4227
            #print(len(test_x[1])) # 6092
            #print(test_x)
           
    #       test_features=utils.get_feature_vectors(test_x)
            #print("test_features")
            #print(len(test_features)) #301
            #print(len(test_features[0])) #10000
            #print(test_features)
            #pred_y=[0.]*len(test_features)
      #      labels = [0.]*len(test_features) #301
            
            #print("len(test_features[0])") #10000 top words
            #print(len(test_features[0]))
        #    for i in range (len(test_features)): #loop through the 301 test_features
                #separate each test_features
            #for feature in test_x[0]:
         #       x=list(test_features[i])
                #print("x as list of features[%d]",i)
                #print(x)
               # x.append(1) #add 1 to the end of list 
                #print("x as list of features[%d] after append",i)
                #print(x)                        
         #       wx_b=sum([self.w[j]*x[j] for j in range(len(self.w))])+self.b
                    
          #      p=0
                
           #     if int(wx_b)>0:
           #         p=1             
            #    else:
          #          p=-1
           #     print("printing p")
            #    print(p)
             #   labels.append(p)
          #  return labels   
        test_features = utils.get_feature_vectors(test_x)
        pred_y = [0.] * len(test_features)
#        print(len(test_features))
        #i = 0
        #for feature in test_features:
        for i in range(len(test_features)):
            #x = list(feature)
            x = test_features[i]
            sumx=sum([test_features[i][j] for j in range(self.dim)] )
            #print("sumx")
            #print(sumx)
#            x.append(1)
#            wx_b = sum([self.weights[j] * x[j] + self.bias[j] for j in range(len(self.weights))])
#            wx_b = wx + self.bias
            wx_b = sum([self.w[j] * x[j]/sumx for j in range(len(self.w))]) + self.b
            #print("in predict")
            #print(int(wx_b))
            if int(wx_b) > 0:
                pred_y[i] = 1
            else:
                pred_y[i] = -1
            #i += 1
#        print(pred_y)
#        print(pred_y.count(-1))
        return pred_y 


class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        self.up=0
        self.learning_step=args.lr
        self.max_iteration=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.b=0
        self.survival=0
                
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
            #TO DO: Learn the parameters from the training data
        
        #train_data[0] features
        #train_data[1] label
        words_bin_form=utils.get_feature_vectors(train_data[0])
        self.w=[0.0]*(len(words_bin_form[0]))
        #self.b=0
        
        #correct_count=0
        time=0
        
        #for i in range len(self.w):
            
        
        
        #print("train_data")
        #print(len(train_data)) #2
        #print(train_data)
        
        #print("train_data[0]") 
        #print(len(train_data[0])) #699
        #print(train_data[0])
        
        #error_found=True
        
        while time<self.max_iteration:  #& error_found:
           # error_found=False
            for i in range(len(words_bin_form)):
                x=words_bin_form[i]
                sumx=sum([words_bin_form[i][j] for j in range(self.dim)] )
                y=train_data[1][i]
            #index=np.random.randint(0,(len(train_data[1])-1))
           # x=list(words_bin_form[index])
            #x.append(1.0)
            #y=train_data[1][index]
            
                wx_b= sum([self.w[j]*x[j]/sumx for j in range(len(self.w))])+self.b
                
            
                si=np.sign(wx_b)
            
            
                time+=1;
            
                if y==si:
                    self.survival+=1
                    break
                else:
                   # error_found=True
                    for j in range(len(self.w)):
                        self.w[j]+=self.learning_step*y*x[j]                       
                        self.survival=1
                        self.b+=self.learning_step*y
                    
            #if wx*y>0:
               # correct_count+=1
                #if correct_count>self.max_iteration:
                  #  break
                #continue    
             
           # for i in range(len(self.w)):
                  #  self.w[i]+=self.learning_step*(y*x[i])
            #print("finishing training perceptron")
            #print("self.w")
            #print(self.w)
            #print("self.b")
            #print(self.b)
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        test_features = utils.get_feature_vectors(test_x)
        pred_y = [0.] * len(test_features)
#        print(len(test_features))
        #i = 0
        #for feature in test_features:
        for i in range(len(test_features)):
            #x = list(feature)
            x = test_features[i]
            sumx=sum([test_features[i][j] for j in range(self.dim)] )
            #print("sumx")
            #print(sumx)
#            x.append(1)
#            wx_b = sum([self.weights[j] * x[j] + self.bias[j] for j in range(len(self.weights))])
#            wx_b = wx + self.bias
            wx_b = sum([self.w[j] * x[j]/sumx for j in range(len(self.w))]) + self.b
            #print("in predict")
            #print(int(wx_b))
            if int(wx_b) > 0:
                pred_y[i] = 1
            else:
                pred_y[i] = -1
            #i += 1
#        print(pred_y)
#        print(pred_y.count(-1))
        return pred_y 
