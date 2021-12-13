'''
Created on 2021年12月11日

@author: Konic
'''
from  dont_patronize_me import DontPatronizeMe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import  pandas as pd
import numpy as np 

dpm=DontPatronizeMe(r'D:\My python\dontpatronizeme_v1.4','dontpatronizeme_pcl.tsv')
dpm.load_task1() 
tfidf=TfidfVectorizer(stop_words='english',max_features=2000)
data=dpm.train_task1_df

# split the dataframe using sample function and using the supletive index set to get the test set 
# train=data.sample(frac=0.8, random_state=42, axis=0)
# test=data[~data.index.isin(train.index)]
data=pd.read_csv(r'D:\My python\train.csv')
test=pd.read_csv(r'D:\My python\test.csv')

# train=data.sample(frac=0.8, random_state=42, axis=0)
# print(len(train))
# trainX,testX,trainY,testY =train_test_split(data['text'],data['label'],test_size=0.2,shuffle=True)




# print(len(pcldf))
# npos=len(pcldf)
# print(npos)
pcldf=data[data.label==1]
npos=len(pcldf)
train=pd.concat([pcldf,data[data.label==0][:npos*2]])
# # 

# 
feature=tfidf.fit_transform(train['text'].astype(str))
# # 
labels=train['label']
# # 
# # print(feature.shape,labels.shape)
# # 
# # trainX,testX,trainY,testY =train_test_split(feature[:8000],labels[:8000],test_size=0.2,shuffle=True)
# # 
# # 
# svm=LinearSVC(C=3)
svm=SGDClassifier(loss='hinge',alpha=1e-3,random_state=50)
# svm=SVC(C=10,kernel='rbf',gamma='scale')
# lr=LogisticRegression()
# nb=MultinomialNB()
# # nb.fit(trainX, trainY)
svm.fit(feature,labels)
# nb.fit(feature,labels )
# # y_pred=lr.predict(testX)

testfeature=tfidf.transform(test.text.astype((str)))
y_pred=svm.predict(testfeature)
# 
# # # print(len(y_pred))
# # acc=np.mean(testY==y_pred)
# # print(acc)
# # right=list(testY==y_pred)
# # 
# # right=(right.count(True))
# # if 1 in y_pred:
# #     print(True)
# # 
f=f1_score(test['label'], y_pred)
# # f1=f1_score(testY, y_pred,average='macro')
print(f)
