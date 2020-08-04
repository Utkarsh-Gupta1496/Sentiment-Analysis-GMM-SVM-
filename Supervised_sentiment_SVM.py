#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:02:06 2019

@author: utkarsh
"""
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

f=open('././movieReviews1000.txt',"r")
#contents is a list with each elements as a line of the text
text=f.readlines()

#removing End Labels
label=[]
for i in range(len(text)):
    label.append((int)(text[i][len(text[i])-2]))
    text[i]=text[i][:(len(text[i])-2)]
    

#word tokenization 
#words=[]  
#for i in text:
##    print(i)
#    words.append(nltk.word_tokenize(i))
#    
#wordset=set()
## all words in all documents
#for i in range(1000):
#    wordset=wordset.union(set(words[i]))
#    
wordfreq = {}
for sentence in text:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
import heapq
most_freq = heapq.nlargest(342, wordfreq, key=wordfreq.get)
sentence_vectors = []
for sentence in text:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
    
sentence_vectors = np.asarray(sentence_vectors)

#tf ifd n=total number of documents
 
N=1000
feture=np.zeros([1000,200])
df=np.sum(sentence_vectors,axis=0)
for i in range(1000):
    for j in range(200):
#        print(np.log(N/df[j]))
        feture[i,j]=(float)(sentence_vectors[i,j]*np.log(N/df[j]))
                
def pca(data,k):
    cov_data=np.cov(np.transpose(data))
    eig_val,eig_vector=np.linalg.eig(cov_data)
    def eigen_sort(value,vector):
        idx = value.argsort()[::-1]   
        eigenValues = value[idx]
        eigenVectors = vector[:,idx]
        return (eigenValues,eigenVectors)
    eig_vals,eig_vectors=eigen_sort(eig_val,eig_vector)
    def final_projection(eigen_matrix,x,k):
        u=eigen_matrix[:,:k]
        y=np.matmul(x,u)
        return y
    x=final_projection(eig_vectors,data,k)
    return x

new_feture=pca(feture,10)
label_train=label[0:700]
label_test=label[700:1000]
train_feature=new_feture[0:700].tolist()
test_feature=new_feture[700:1000].tolist()

y,x=label_train,train_feature
prob  = svm_problem(y, x)
#Kernal=linear
param = svm_parameter('-t 3 -c 2 -b 0')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(label_test, test_feature, m)
