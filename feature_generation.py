# -*- coding: utf-8 -*- 
from __future__ import print_function
#coding:utf-8

import numpy as np 
import random
import tensorflow as tf
from nets import nets_factory
from nets import resnet_v1
from preprocessing import preprocessing_factory
import os
import imtools
import tensorflow as tf
import semantic_embedding
import csv
import utils
import scipy.misc as cv2
from random import shuffle
import pickle as cPickle
from PIL import Image
from sklearn import preprocessing
import vgg
import matplotlib
import bhtsne
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
import matplotlib.image as mpimg # mpimg 用于读取图片
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#######################################################
#使用预训练的VGG网络，提取画作的内容特征（最后一层卷积层），pca降维1024d
#得到一个kpl文件，画作特征以及对应的图片路径
#单个画家计算，要改画家路径等
#######################################################



def get_visual_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            "vgg_16",
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            "vgg_16",
            is_training=False)

        size = 256
        img_bytes = tf.read_file(FLAGS)
     
        image = tf.image.decode_jpeg(img_bytes,channels=3)
        # image = _aspect_preserving_resize(image, size)
        images = tf.stack([image_preprocessing_fn(image, size, size)])
        #images= preprocessing.normalize(images, norm='l2')
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        
        visual_layers=["vgg_16/conv5/conv5_1"]
        for layer in visual_layers:
            features = endpoints_dict[layer]

        with tf.Session() as sess:
            init_func = utils._get_init_fn()
            init_func(sess)
         
            features=sess.run(features)
        features=np.ndarray.flatten(np.array(features))        
        #for n in xrange(512):#change the input shape to 512*1024
        #    features=np.array(features)
        #    img_features.append(np.ndarray.flatten(features[:,:,:,n])) 
        return features

    

def main():

    csvFile = open("list500_5.csv", "r")   ##23 artist
    #csvFile = open("100_2.csv", "r")      ##208 artist
    reader = csv.reader(csvFile)
    
    artist=[]
    
    
    for item in reader:
        if reader.line_num == 1:
            continue
        artist.append(item[1])
    csvFile.close()
    artist=list(set(artist))
    print(len(artist))
    print(artist)
    test_name_path=[]
    painting_content=[]
    k=0
    path="../Images"
    fs = os.listdir(path) 
  
    for f1 in artist:
    #f1="pyotr-konchalovsky"
        tmp_path = os.path.join(path,f1)  
        fs_images= os.listdir(tmp_path)  
        for i in range(int(len(fs_images))):
            #print(i)
            #if (fs_images[i] in painting) and (painting_artist[fs_images[i]]==f1):
            
            k=k+1
            print("------------",k)
            test_name_path.append(tmp_path+"/"+fs_images[i])
            #painting_name.append(fs_images[i])
            print(tmp_path+"/"+fs_images[i])
            painting_content.append(get_visual_features(tmp_path+"/"+fs_images[i]))
                
                    #img_bytes = tf.read_file(tmp_path+"/"+fs_images[i])
                    #image = tf.image.decode_jpeg(img_bytes,channels=3)

        print(len(test_name_path))
    
    # write_file=open("kpl/content_before_23.kpl",'wb')
    # cPickle.dump(painting_content,write_file,-1)
    # cPickle.dump(test_name_path,write_file,-1)
    # write_file.close()  

    # transformer = IncrementalPCA(n_components=1024, batch_size=20)
    # #transformer = PCA(n_components=1024)
    # new_feature=transformer.fit_transform(np.array(painting_content))
    svd = TruncatedSVD(n_components=1024, random_state=42)
    new_feature=svd.fit_transform(np.array(painting_content))
  
    ##写数据
    write_file=open("kpl/content_23.kpl",'wb')
    cPickle.dump(new_feature,write_file,-1)
    cPickle.dump(test_name_path,write_file,-1)

    write_file.close()  

        
if __name__ == '__main__':
    main()
