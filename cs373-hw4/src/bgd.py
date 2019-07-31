from __future__ import division
from classifier import BinaryClassifier
from utils import get_feature_vectors
from random import seed
from random import random

import numpy.matlib
import numpy as np
from numpy import linalg as LA
import pandas as pd

import utils
import math
import random


class BGDHinge(BinaryClassifier):

    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        
        self.lr=args.lr_bgd
        self.mi=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.gw=None
        self.b=None
        self.gb=None
        self.l=None
        self.wx_b=None
        self.binFeats=args.bin_feats
    



    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        x,y=train_data
        words_bin_form=np.asarray(utils.get_feature_vectors(x,self.binFeats))
        y=np.asarray(y)
        l=len(x) #699
        
        self.w=[0.0]*(len(words_bin_form[0]))
        self.b=0
       
        
        for x in range(0, self.mi):
            
            
            
            for i in range(l):#699
                dg=[0.0]*(len(words_bin_form[0]))
                db=0
                label=y[i]
                single=words_bin_form[i]                
                if label*(np.dot(self.w,single)+self.b)<=1:
                        dg=np.multiply(-1*label,single)
                        db=-1*label
                        if LA.norm(dg)<0.00001:
                              break
            self.w=np.subtract(self.w,np.multiply(self.lr,dg))
            self.b-=db*self.lr                               
                     
                          
               
            
          
         
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
       
        words_bin_form=utils.get_feature_vectors(test_x,self.binFeats)
        result=[]
        for line in words_bin_form:
            if np.dot(line,self.w)+self.b>=0:
                result.append(1)
            else:
                result.append(-1)
        return result    
       
    

class BGDLog(BinaryClassifier):
    def sigmoid(self, x, weight, bias):
        z=np.dot(weight,x)+bias
        if z>700:
            z=700
        if z<-700:
            z=-700
    
        return 1.0 / (1.0 + np.exp(-1*z))    
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        self.up=0
        self.lr=args.lr_bgd
        self.mi=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.b=None
        self.l=None
        self.wx_b=None
        self.la=args.la
        self.binFeats=args.bin_feats
        self.vocab_size=args.vocab_size
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        x,y=train_data
        words_bin_form=np.asarray(utils.get_feature_vectors(x,self.binFeats))
        y=np.asarray(y)
        l=len(x) #699
        
        self.w=[0.0]*(self.vocab_size)
        self.b=0
       
        
        for x in range(0, self.mi):
            
            dg=[0.0]*(self.vocab_size)
            db=0
            
            for i in range(l):#699
                label=y[i]
                if label==-1:
                    label=0
                single=words_bin_form[i]
                sig=self.sigmoid(single, self.w, self.b)
                dg=np.subtract(dg, np.multiply( label - sig, single))
                db+=sig - label
            self.w=np.subtract(self.w, np.multiply(self.lr/l, dg))
            self.b-=db*self.lr/l
         
            t=LA.norm(dg)
            if t<0.00001:
                break
            
          
         
    def predict(self, test_x):
      
        #TO DO: Compute and return the output for the given test inputs
       
        words_bin_form=utils.get_feature_vectors(test_x,self.binFeats)
        result=[]
        for line in words_bin_form:
            if self.sigmoid(line, self.w, self.b) >=0.5:
                result.append(1)
            else:
                result.append(-1)
        return result
    
         
class BGDHingeReg(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        self.up=0
        self.lr=args.lr_bgd
        self.mi=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.b=None
        self.l=None
        self.wx_b=None
        self.la=args.la
        self.binFeats=args.bin_feats
        
    def fit(self, train_data):
        x,y=train_data
        words_bin_form=np.asarray(utils.get_feature_vectors(x,self.binFeats))
        y=np.asarray(y)
        l=len(x) #699
        
        self.w=[0.0]*(len(words_bin_form[0]))
        self.b=0
       
        
        for x in range(0, self.mi):
            
            
            
            for i in range(l):#699
                dg=[0.0]*(len(words_bin_form[0]))
                db=0
                label=y[i]
                single=words_bin_form[i]                
                if label*(np.dot(self.w,single)+self.b)<=1:
                        dg=np.multiply(-1*label,single)
                        db=-1*label
                        if LA.norm(dg)<0.00001:
                              break
            self.w=np.subtract(self.w,np.multiply(self.lr,dg))
            self.b-=db*self.lr                               
                     
                          
               
            
          
         
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        words_bin_form=utils.get_feature_vectors(test_x,self.binFeats)
        result=[]
        for line in words_bin_form:
            if np.dot(line, self.w)+self.b >=0:
                result.append(1)
            else:
                result.append(-1)
        return result
    
        
class BGDLogReg(BinaryClassifier):
    def sigmoid(self, x, weight, bias):
        z=np.dot(weight,x)+bias
        if z>700:
            z=700
        if z<-700:
            z=-700  
        return 1.0 / (1.0 + np.exp(-1*z))      
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args=args
        self.up=0
        self.lr=args.lr_bgd
        self.mi=args.num_iter
        self.dim=args.f_dim
        self.w=None
        self.b=None
        self.l=None
        self.wx_b=None
        self.binFeats=args.bin_feats
        self.vocab_size=args.vocab_size
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        x,y=train_data
        words_bin_form=np.asarray(utils.get_feature_vectors(x,self.binFeats))
        y=np.asarray(y)
        l=len(x) #699
        
        self.w=[0.0]*(self.vocab_size)
        self.b=0
       
        
        for x in range(0, self.mi):
            
            dg=[0.0]*(len(words_bin_form[0]))
            db=0
            
            for i in range(l):#699
                label=y[i]
                if label==-1:
                    label=0
                single=words_bin_form[i]
                sig=int(self.sigmoid(single, self.w, self.b))
                dg=np.subtract(dg, np.multiply( label - sig, single))
                db+=sig - label
            self.w=np.subtract(self.w,np.multiply(self.lr/l,dg))
            self.b-=db*self.lr/l
         
            t=LA.norm(dg)
            if t<0.00001:
                      break
            
           
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        words_bin_form=utils.get_feature_vectors(test_x,self.binFeats)
        result=[]
        for line in words_bin_form:
            if self.sigmoid(line, self.w, self.b) >=0.5:
                result.append(1)
            else:
                result.append(-1)
        return result
    
    


