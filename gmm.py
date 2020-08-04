"""
Created on Tue Sep 17 10:02:06 2019

@author: utkarsh
"""

import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

f=open('./data/movieReviews1000.txt',"r")
#contents is a list with each elements as a line of the text
text=f.readlines()

#removing End Labels
label=[]
for i in range(len(text)):
    label.append((int)(text[i][len(text[i])-2]))
    text[i]=text[i][:(len(text[i])-2)]
    

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


#now applying GMM(2 mixture)
#finding initial initialization 
kmeans = KMeans(n_clusters=2, random_state=0).fit(new_feture)
labels=kmeans.labels_
cluster_centers=kmeans.cluster_centers_


mean1=cluster_centers[0,:]
mean2=cluster_centers[1,:]
mean_list=[mean1,mean2]

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

data1=np.transpose(new_feture[ClusterIndicesNumpy(0,kmeans.labels_)])
data2=np.transpose(new_feture[ClusterIndicesNumpy(1,kmeans.labels_)])
label=np.array(label)
#t data1 has 0 review
#t data2 has 1 review 
tdata1=np.transpose(new_feture[ClusterIndicesNumpy(0,label)])
tdata2=np.transpose(new_feture[ClusterIndicesNumpy(1,label)])

data1=(np.array(data1))
data2=(np.array(data2))
plt.figure()
#plt.scatter(data1[0,:],data1[1,:],c='c',label='Kmean',marker='o')
#plt.scatter(data2[0,:],data2[1,:],c='m',label='Kmean',marker='o')
#plt.grid(True)
#plt.legend(loc='upper right')
cov1=np.cov(data1)
cov2=np.cov(data2)

tdata1=np.transpose(tdata1)
tdata2=np.transpose(tdata2)
#


cov_list=[cov1,cov2]

alpha1=data1.shape[1]/new_feture.shape[0]
alpha2=1-alpha1

alpha_list=[alpha1,alpha2]
def liklihood(data_array_transposed,alpha_list,cov_list,mean_list):
    j=len(alpha_list)
    n=data_array_transposed.shape[1]
    weighted_normal=[]
    like=0
    for i in range(n):
        for k in range(j):
            weighted_normal.append(alpha_list[k]*(multivariate_normal.pdf(data_array_transposed[:,i],mean=mean_list[k],cov=cov_list[k])))
        a=np.array(weighted_normal)
        l=np.log(np.sum(a))
        weighted_normal=[]
        like=like+l
    return like

cov1d=np.diag(np.diag(cov1))
cov2d=np.diag(np.diag(cov2))
covd_list=[cov1d,cov2d]
L_initial=liklihood(np.transpose(new_feture),alpha_list,covd_list,mean_list)



def maximization(data_array_transposed,alpha_list,cov_list,mean_list,f):
    #if f=1 then we find diagonal covariance
    #calculation of gaama ij
    #calculation of Gamma(dinominator)
    j=len(alpha_list)
    n=data_array_transposed.shape[1]
    m=data_array_transposed.shape[0]
    weighted_normal=np.zeros((n,j))
    gama=np.zeros((n,j))
    for i in range(n):
        for k in range(j):
            x=(alpha_list[k]*(multivariate_normal.pdf(data_array_transposed[:,i],mean=mean_list[k],cov=cov_list[k])))
            weighted_normal[i,k]=x
    dinominator=np.sum(weighted_normal,axis=1) 
    for i in range(n):
        for k in range(j):
            gama[i,k]=weighted_normal[i,k]/dinominator[i]
    mean_deviation_dinomintor=np.sum(gama,axis=0)
    alpha=mean_deviation_dinomintor/n
    #updated weights
    alpha_list_updated=alpha.tolist()
    
#mean calculation
    #numerator
    sum_mean=[]
    for k in range(j):
        e=np.zeros(m)
        for i in range(n):
            v=(float)(gama[i,k])
            if e.shape==data_array_transposed[0:m,i].shape:
                
                e=e+v*data_array_transposed[:,i]
            else:
                print('gfs')
        sum_mean.append(e)
    mean_sum=np.array(sum_mean)
    mean_new=[]
    for k in range(j):
        mean_new.append(mean_sum[k]/mean_deviation_dinomintor[k])
    
#standard dev
    sum_mean1=[]
    for k in range(j):
        e=np.zeros([m,m])
        for i in range(n):
            v=(float)(gama[i,k])
            
                
            e=e+v*np.outer(data_array_transposed[:,i]-mean_new[k],data_array_transposed[:,i]-mean_new[k])
            
        sum_mean1.append(e)
    mean_sum1=np.array(sum_mean1)
    sd_new=[]
    for k in range(j):
        if(f==1):
            sd_new.append(np.diag(np.diag(mean_sum1[k]/mean_deviation_dinomintor[k])))
        else:
            sd_new.append(mean_sum1[k]/mean_deviation_dinomintor[k])
        
        
            
           
    return sd_new,mean_new,alpha_list_updated
def classification(sigma,alpha,mean_,data):
    n=data.shape[0]
    data_transposed=np.transpose(data)
    data0=[]
    data1=[]
    for i in range(n):
        w0=multivariate_normal.pdf(data_transposed[:,i],mean=mean_[0],cov=sigma[0])
        w1=multivariate_normal.pdf(data_transposed[:,i],mean=mean_[1],cov=sigma[1])
        if np.log(w0)>=np.log(w1):
            data0.append(data_transposed[:,i])
        else:
            data1.append(data_transposed[:,i])
    
    return data0,data1
    
def plot(sigma,alpha,mean_,data,j):
    data_transposed=np.transpose(data)
    data1=[]
    data2=[]
    for i in range(1000):
        w1=multivariate_normal.pdf(data_transposed[:,i],mean=mean_[0],cov=sigma[0])
        w2=multivariate_normal.pdf(data_transposed[:,i],mean=mean_[1],cov=sigma[1])
        if np.log(w1)>=np.log(w2):
            data1.append(data_transposed[:,i])
        else:
            data2.append(data_transposed[:,i])
    s="EM Iteration ="+str(j)
    data1=np.array(data1)
    data2=np.array(data2)
    plt.figure()
    plt.scatter(data1[:,0],data1[:,1],c='c',marker='o',label=s)
    plt.scatter(data2[:,0],data2[:,1],c='m',marker='o',label=s)
    plt.scatter(mean_[0][0],mean_[0][1],c='k',marker='D',label="Mean(Blue Dots)",s=100)
    plt.scatter(mean_[1][0],mean_[1][1],c='y',marker='D',label="Mean(Purple Dots)",s=100)
    plt.grid(True)
    plt.legend(loc='upper right')
    
      


#    

i=1
iteration_diagonal=[1]
liklihood_diagonal=[L_initial]
k=[1]
sd_updated,mean_updated,alpha_list_updated1=maximization(np.transpose(new_feture),alpha_list,covd_list,mean_list,1)

#training Speech Data + full covariance flag =0
sigma_diag=[]
mean_diag=[]
alpha_diag=[]
while(1):
    check=0
    sd_updated,mean_updated,alpha_list_updated1=maximization(np.transpose(new_feture),alpha_list_updated1,sd_updated,mean_updated,1)
    L1=liklihood(np.transpose(new_feture),alpha_list_updated1,sd_updated,mean_updated)
    plot(sd_updated,alpha_list_updated1,mean_updated,new_feture,i)
    i=i+1
    iteration_diagonal.append(i)
    liklihood_diagonal.append(L1)
    diff=(liklihood_diagonal[i-1]-liklihood_diagonal[i-2])
    if i>=10:
        for x in range(i,i-10,-1):
            diff1=(liklihood_diagonal[x-1]-liklihood_diagonal[x-2])
            if diff1<=1:
                check=check+1
        if check==9:
            sigma_diag=sd_updated
            mean_diag=mean_updated
            alpha_diag=alpha_list_updated1
            break
        else:
            check=0
    if i>=60:
        sigma_diag=sd_updated
        mean_diag=mean_updated
        alpha_diag=alpha_list_updated1
        break


s="Number of Iteration ="+str(i-1)
plt.figure()
plt.plot(iteration_diagonal,liklihood_diagonal,marker='*',c='g',label=s)
plt.xlabel("Iteration")
plt.ylabel("Log Liklihood")
plt.legend(loc='upper right')
plt.title("Number Of Gaussian=2,Train Data,Diagonal Covariance")
plt.show()

#classification
c1,c2=classification(sd_updated,alpha_list_updated1,mean_updated,tdata1)
k1,k2=classification(sd_updated,alpha_list_updated1,mean_updated,tdata2)

accuracy=(len(c2)+len(k1))/1000
print("The classification Accuracy is {}".format (accuracy))


        
