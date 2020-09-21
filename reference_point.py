# -*- coding: utf-8 -*- 
from __future__ import print_function
#coding:utf-8

import numpy as np 
import random
import tensorflow as tf

from preprocessing import preprocessing_factory
import os
import imtools
import tensorflow as tf

import csv

from random import shuffle
import pickle as cPickle
from PIL import Image
from sklearn import preprocessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
import matplotlib.image as mpimg # mpimg 用于读取图片
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
#######################################################################
###计算没有代表性数值的画作同类别里最近的n个有代表性数值的点
#######################################################################


#csvFile = open("list500_5.csv", "r")
csvFile = open("100_2.csv", "r")
reader = csv.reader(csvFile)
painting=[]
artist=[]
test_name=[]
painting_class={}
painting_style={}
painting_styleclass={}
name_code={}
i=-1
for item in reader:
    if reader.line_num == 1:
        continue
    painting.append("../Images/"+item[1]+"/"+item[2])  
    painting_class["../Images/"+item[1]+"/"+item[2]]=item[3] 
    painting_styleclass["../Images/"+item[1]+"/"+item[2]]=item[5] 
    #painting_style[item[2]]=item[4]
    artist.append(item[1])
csvFile.close()
print(len(painting))
def cal_n(lab,labs,image_name_score,inX,i,image_name_nonscore,dataSet, k):
    print(np.array(dataSet).shape)
    k_n_image=[]
    k_n_dis=[]
    k_n_lab=[]
    k_n_feature=[]
    dataSetSize = np.array(dataSet).shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5 #计算欧式距离
    #print(distances)
    sortedDistIndicies= np.argsort(distances)#排序并返回index
    sortedDist=sorted(distances,reverse=False)


    #选择距离最近的k个同类值 5个不同类
    #print(sortedDistIndicies)
    if image_name_nonscore[i] in painting:
        for j in range(len(labs)):
    	#print(sortedDist[j])
            #if labs[sortedDistIndicies[j]] == lab and len(k_n_image) < 15:
            if distances[sortedDistIndicies[j]] != 0 and len(k_n_image) < 30:
                if image_name_score[sortedDistIndicies[j]] in painting:
                    k_n_image.append(image_name_score[sortedDistIndicies[j]])
                    k_n_feature.append(dataSet[sortedDistIndicies[j]])
                    k_n_dis.append(distances[sortedDistIndicies[j]])
                    k_n_lab.append(painting_class[image_name_score[sortedDistIndicies[j]]])
            #print(painting_class[image_name_score[sortedDistIndicies[j]]])
            #print(labs[sortedDistIndicies[j]])
    # for j in range(1000):
    #     #print(sortedDist[j])
    #     if labs[sortedDistIndicies[j]] != lab and len(k_n_image) < 15:
    #         k_n_image.append(image_name_score[sortedDistIndicies[j]])
    #         k_n_feature.append(dataSet[sortedDistIndicies[j]])
    #         k_n_dis.append(distances[sortedDistIndicies[j]])
    #         k_n_lab.append(painting_class[image_name_score[sortedDistIndicies[j]]])
        print(image_name_nonscore[i],painting_class[image_name_nonscore[i]])
        print(k_n_image)
        print(k_n_dis)
    print(k_n_lab)
    print('--------------------------------------')
    return k_n_feature,k_n_image,k_n_dis,k_n_lab



#读数据
with open("kpl/208features_after_reduce.kpl", 'rb') as f:
    features = cPickle.load(f, encoding='latin1')
    #total_features1 = preprocessing.normalize(total_features1, norm='l2')
    image_name = cPickle.load(f, encoding='latin1')
    image_lable = cPickle.load(f, encoding='latin1')
# read_file=open("kpl/features208_after_reduce.kpl",'rb')
# features=cPickle.load(read_file)
# image_name=cPickle.load(read_file)
# image_lable=cPickle.load(read_file)
for i in range(len(image_name)):
    #print(i,image_name[i])
    image_name[i]=image_name[i].decode("utf-8")
num=len(image_name)
x_min, x_max = np.min(features, 0), np.max(features, 0)
#print(x_min)
features = (features - x_min) / (x_max - x_min)
#score1= os.listdir("score208/") 
csvFile = open("score208.csv", "r")
reader = csv.reader(csvFile)
score1=[]
for item in reader:
    if reader.line_num == 1:
        continue
    score1.append("../Images/"+item[1]+"/"+item[2])
csvFile.close()
feature_socre=[]
feature_nonsocre=[]
image_name_score=[]
image_name_nonscore=[]
image_lable_nonscore=[]
image_lable_score=[]
for i in range(num):
    print(image_name[i])
    if image_name[i] in score1:
        feature_socre.append(features[i])
        image_name_score.append(image_name[i])
        image_lable_score.append(image_lable[i])
        #有标签的特征和图片名
    else:
        feature_nonsocre.append(features[i])
        image_name_nonscore.append(image_name[i])
        image_lable_nonscore.append(image_lable[i])
        #无标签的特征和图片名

k_n_image_collection=[]
k_n_dis_collection=[]
k_n_lab_collection=[]
k_n_feature_collection=[]
for i in range(len(feature_socre)):
#for i in xrange(10):

    k_n_feature,k_n_image,k_n_dis,k_n_lab=cal_n(image_lable_score[i],image_lable_score,
        image_name_score,feature_nonsocre[i],i,image_name_nonscore,feature_socre,10) 
    k_n_image_collection.append(k_n_image)
    k_n_dis_collection.append(k_n_dis)
    k_n_lab_collection.append(k_n_lab)
    k_n_feature_collection.append(k_n_feature) 
    
for i in range(len(feature_nonsocre)):
#for i in xrange(10):

    k_n_feature,k_n_image,k_n_dis,k_n_lab=cal_n(image_lable_nonscore[i],image_lable_score,
        image_name_score,feature_nonsocre[i],i,image_name_nonscore,feature_socre,10) 
    k_n_image_collection.append(k_n_image)
    k_n_dis_collection.append(k_n_dis)
    k_n_lab_collection.append(k_n_lab)
    k_n_feature_collection.append(k_n_feature) 


##写数据
write_file=open("kpl/reference_point30_withscore.kpl",'wb')
cPickle.dump(feature_nonsocre,write_file,-1)            #没有代表性的画作特征
cPickle.dump(image_name_nonscore,write_file,-1)         #没有代表性的画作名
cPickle.dump(image_lable_nonscore,write_file,-1)        #没有代表性的画作类别
cPickle.dump(k_n_feature_collection,write_file,-1)      #没有代表性的画作周围k个有代表性的画作特征
cPickle.dump(k_n_image_collection,write_file,-1)        #没有代表性的画作周围k个有代表性的画作名
cPickle.dump(k_n_dis_collection,write_file,-1)          #没有代表性的画作周围k个有代表性的画作与该画作的距离
cPickle.dump(k_n_lab_collection,write_file,-1)          #没有代表性的画作周围k个有代表性的画作类别
write_file.close()              
#     n_total=n_total+n
# n_total=n_total/float(len(feature_nonsocre))
#print(n_total)