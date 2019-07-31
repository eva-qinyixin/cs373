#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

Author: Rajkumar Pujari
Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
import utils
import numpy as np

class NaiveBayes(BinaryClassifier):

	def __init__(self, args):
		#TO DO: Initialize parameters here
		#raise NotImplementedError
		self.args=args
		self.pos_words=None
		self.neg_words=None
		self.prediction=None        

	def fit(self, train_data):
		#TO DO: Learn the parameters from the training data
		# find all words positive weight and negative weight
		# p(c+|word) p(c-|word)
		# wfr:word's frequency ratio 
		# accuracy
		
		#two classes:positive and negative
		#a data instance is a movie review d 
		#which is a sequence of words w1,w2,..,wn 
		#idea:estimate the probability of each class for a given instnac		#e
		#estimate P(c+|d)and P(c-|d). assign the class with higher proba		#bility score
		#how to estimate the probabilities

		#using Bayes theorem , P(c+|d)=P(d|c+)P(c+)/P(d)
		#we can drop the denominator 
		#as we are only interested in comparing P(c+|d) to P(c-|d)
		#P(c+) can be learnt from the trainng data as 
		#size(c+)/(size(c+)+size(c-))
		#to estimate P(d|c+)
		#P(d|c+)=P(w1w2...wn|c+)
		#learn P(wi|c+) and P(wi|c-) from the training data

		#use for loop to seperate c+ and c-
		#wi the frequency of showing up of the words
		# 100000 words positive and negative
		


		words_freq_form=utils.get_feature_vectors(train_data[0])
		#print("words_freq[0]:%d",len(words_freq_form[0])) 
    
		#print(words_freq_form)  #699 rows each with  10000 popular words fre
		#num_words=sum([sum(i) for i in zip(*words_freq_form)])
		#print("num_words: %d",num_words)
		self.pos_words=[0.]*(len(words_freq_form[0]))
		self.neg_words=[0.]*(len(words_freq_form[0]))
		
		#print(len(words_freq_form[0]))#10000
		#print(len(words_freq_form)) #699
		#print(train_data)

		#print(len(train_data)) #2 row
		#print(len(train_data[0])) #699 columns
		#print("train_data[1]")
		#print(train_data[1])
		for i in range(0,(len(words_freq_form[0]))):
		   #loop for each positive word
			pos_word=0.0
			neg_word=0.0
			
			for j in range(0, (len(words_freq_form))):
				#loop through frequency for each word
				#word_freq_form[j][i]
				if train_data[1][j]==1:
					pos_word=pos_word+words_freq_form[j][i]
				elif train_data[1][j]==-1:
					neg_word=neg_word+words_freq_form[j][i]
                    
		   	#update the frequency for the word both for pos and neg
			self.pos_words[i]=pos_word
			self.neg_words[i]=neg_word
		#print("pos_words")
		#print(pos_words)
		#print("neg_words")
		#print(neg_words) 
 			       
			#if(words_freq_form[i]) 				
		#print(train_data[1])
		#y=np.array(train_data[1])
		#print(len(words_freq))
		#print(np.count_nonzero(y == 1))
		#raise NotImplementedError
        

	def predict(self, test_x):
		#TO DO: Compute and return the output for the given test inputs
		#print("test")
		#print(len(test_x))
		#print(test_x)
		#print("test_x[0]")
		#print(len(test_x[0]))
		#print(test_x[0])
		#print("test_x[1]")
		#print(len(test_x[1]))
		#print(test_x[1])
		#print("predict")
		words_freq_form_test=utils.get_feature_vectors(test_x)
		self.prediction=[0.]*(len(test_x))        
		for i in range(0, len(test_x)):    
				sum=0.0            
				for j in range(0, len(words_freq_form_test[i])): #every word
						sum+=self.pos_words[j]/(self.pos_words[j]+self.neg_words[j])*words_freq_form_test[i][j]  
						sum-=self.neg_words[j]/(self.pos_words[j]+self.neg_words[j])*words_freq_form_test[i][j]
				if sum>0:                        
						self.prediction[i]=1
				elif sum<0:
						self.prediction[i]= -1
				elif sum==0:
						self.prediction[i]= 0
		return self.prediction
		#clprint(words_freq_form)
