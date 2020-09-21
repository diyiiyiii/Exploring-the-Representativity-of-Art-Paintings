# -*- coding: utf-8 -*- 
from __future__ import print_function
#coding:utf-8
from sklearn import datasets
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
from random import shuffle
import pickle as cPickle
from PIL import Image
from sklearn import preprocessing
import vgg

# from skimage import io
# from skimage import feature as ft
# from sklearn.decomposition import PCA 
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

lable_size=23
style_size=14
training_epochs=40
batch_size=32
learning_rate=1e-3
NUM_CLASSES=251
IMAGE_SIZE=224
train_log_file="resnet50.ckpt"
#csvFile = open("list.csv", "r")
train_name_path=[]
train_lable=[]
artist=[]
test_name_path=[]
test_lable=[]

train_style=[]
test_style=[]





def relu(x, alpha=0.02, max_value=None):
    x = tf.maximum(alpha*x,x)
    return x 
def preprocess(name_path_batch):
    images=[]
    
    size = 224
    for j in range(batch_size):

        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                            "vgg_16",
                            is_training=True)  #return preprocessing fuction
     
        img_bytes = tf.read_file(name_path_batch[j])
    #img_bytes= Image.open(FLAGS)
        #if name_path_batch[j].lower().endswith('png'):
        #    image = tf.image.decode_png(img_bytes,channels=3)
        #else:
        image = tf.image.decode_jpeg(img_bytes,channels=3)
        images.append(image)
    
    images_ = tf.stack([image_preprocessing_fn(image, size, size) for image in images])  
    return images_ 

def main():
#####################################################
#生成batch
#####################################################
    #csvFile = open("100_2.csv", "r")
    csvFile = open("list500_5.csv", "r")
    reader = csv.reader(csvFile)
    painting=[]
    artist=[]
    painting_class={}
    painting_style={}
    painting_styleclass={}
    name_code={}
    i=-1
    for item in reader:
        if reader.line_num == 1:
            continue
        painting.append(item[1]+item[2])  
        painting_class[item[1]+item[2]]=item[3] 
        painting_styleclass[item[1]+item[2]]=item[5] 
        #painting_style[item[2]]=item[4]
        artist.append(item[1])
    csvFile.close()
    # read_file=open("style_c.pkl",'rb')   
    # style_c_file=cPickle.load(read_file)
    # read_file.close()

    path="../Images"
    fs = os.listdir(path) 
    for f1 in fs:     
        if f1 in artist:
            tmp_path = os.path.join(path,f1)  
            fs_images= os.listdir(tmp_path)
            fs_images.sort()
            for i in range(len(fs_images)):
                if i <len(fs_images)*0.86:
                    if  f1+fs_images[i] in painting:
                        train_name_path.append(tmp_path+"/"+fs_images[i])   
                        #train_style.append(style_c_file[painting_style[fs_images[i]]])  
                        train_style.append(int(painting_styleclass[f1+fs_images[i]]))         
                        train_lable.append(int(painting_class[f1+fs_images[i]]))
                else:
                    if  f1+fs_images[i] in painting:
                        test_name_path.append(tmp_path+"/"+fs_images[i])
                        #test_style.append(style_c_file[painting_style[fs_images[i]]])
                        test_style.append(int(painting_styleclass[f1+fs_images[i]])) 
                        test_lable.append(int(painting_class[f1+fs_images[i]]))

    
    a=tf.cast(train_name_path,tf.string)
    b=tf.cast(train_lable,tf.int32)
    c=tf.cast(np.array(train_style),tf.int32)
    train_input_queue = tf.train.slice_input_producer([a,b,c],shuffle=True,num_epochs=training_epochs)   
    train_name_path_batch,train_lable_batch,train_style_batch = tf.train.batch(train_input_queue,
                                                 batch_size = batch_size,
                                                 num_threads = 40,
                                                 capacity = 100)
    train_name_path_batch=preprocess(train_name_path_batch)
    train_lable_batch=tf.one_hot(train_lable_batch,lable_size)
    train_style_batch=tf.one_hot(train_style_batch,style_size)

    a1=tf.cast(test_name_path,tf.string)
    b1=tf.cast(test_lable,tf.int32)
    c1=tf.cast(np.array(test_style),tf.int32)
    test_input_queue = tf.train.slice_input_producer([a1,b1,c1],shuffle=True,num_epochs=training_epochs)   
    test_name_path_batch,test_lable_batch,test_style_batch = tf.train.batch(test_input_queue,
                                                 batch_size = batch_size,
                                                 num_threads = 40,
                                                 capacity = 100)
    test_name_path_batch=preprocess(test_name_path_batch)
    test_lable_batch=tf.one_hot(test_lable_batch,lable_size)
    test_style_batch=tf.one_hot(test_style_batch,style_size)
    train_num=len(train_name_path)
    test_num=len(test_name_path)

    train_num=len(train_name_path)
    test_num=len(test_name_path)

    
    #用于保存微调后的检查点文件和日志文件路径
    train_log_dir = 'model/resnet_unify23_new'    
    
    #官方下载的检查点文件路径
    checkpoint_file = 'pretrained/resnet_v1_50.ckpt'
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)

 

    arg_scope =resnet_v1.resnet_arg_scope()
    print('arg_scope', arg_scope)
    #创建网络
    with  slim.arg_scope(arg_scope):
        
        '''
        2.定义占位符和网络结构
        '''        
        #输入图片
        input_images = tf.placeholder(dtype=tf.float32,shape = [None,224,224,3],name="input_images")
        #图片标签
        input_style= tf.placeholder(dtype=tf.float32,shape = [None,style_size],name="input_style")
        input_labels = tf.placeholder(dtype=tf.float32,shape = [None,lable_size],name="input_labels")        
        #训练还是测试？测试的时候弃权参数会设置为1.0
        is_training = tf.placeholder(dtype = tf.bool,name="is_training")
        
        #创建vgg16网络  如果想冻结所有层，可以指定slim.conv2d中的 trainable=False
        net1,end_points =  resnet_v1.resnet_v1_50(input_images, is_training=is_training,num_classes = lable_size)
        
        params = slim.get_variables_to_restore(exclude=['resnet_v1_50/logits'])
        #用于恢复模型  如果使用这个保存或者恢复的话，只会保存或者恢复指定的变量
        restorer = tf.train.Saver(params) 

        Weights1 = tf.Variable(tf.truncated_normal([style_size, 256],mean=0.0,stddev=1.0,dtype=tf.float32))  
        biases1 = tf.Variable(tf.zeros([batch_size, 256],dtype=tf.float32) + 0.1)
        net2 = tf.matmul(input_style, Weights1,transpose_a=False) + biases1
        net2 = relu(net2)
        tf.add_to_collection('pred_network',net2)
        
        Weights2 = tf.Variable(tf.truncated_normal([256, 128],mean=0.0,stddev=1.0,dtype=tf.float32))  
        biases2 = tf.Variable(tf.zeros([batch_size, 128],dtype=tf.float32) + 0.1)
        net2 = relu(tf.matmul(net2, Weights2,transpose_a=False) + biases2)

        net1 = tf.squeeze(net1, [1, 2], name='squeezed')   
        feature= tf.concat([net1,net2],1)         
        net1=tf.nn.l2_normalize(net1,dim=1)
        net2=tf.nn.l2_normalize(net2,dim=1)

        print(net1.get_shape(),net2.get_shape())
        net=tf.concat([net1,net2],1)
        tf.add_to_collection('pred_network',net)
        
        Weights3 = tf.Variable(tf.truncated_normal([2176, lable_size],mean=0.0,stddev=1.0,dtype=tf.float32))  
        biases3 = tf.Variable(tf.zeros([batch_size, lable_size],dtype=tf.float32) + 0.1)
        logits = relu(tf.matmul(net, Weights3,transpose_a=False) + biases3)
        predictions = tf.nn.softmax(logits)

        #预测标签
        pred = tf.argmax(predictions,axis=1)

        '''
        3 定义代价函数和优化器
        '''                
        #代价函数
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels,logits=logits))
        
        #设置优化器
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
        #预测结果评估        
        correct = tf.equal(pred,tf.argmax(input_labels,1))                    #返回一个数组 表示统计预测正确或者错误 
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))                #求准确率
        tf.add_to_collection('pred_network',accuracy)                    
        #num_batch = int(np.ceil(n_train / batch_size))
        
        #用于保存检查点文件 
        save = tf.train.Saver(write_version=tf.train.SaverDef.V1) 
        #恢复模型
        with tf.Session() as sess:
            init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            sess.run(init_op)
            
                    
            #检查最近的检查点文件
            ckpt = tf.train.latest_checkpoint(train_log_dir)
            if ckpt != None:
                save.restore(sess,ckpt)
                print('从上次训练保存后的模型继续训练！')
            else:
                restorer.restore(sess, checkpoint_file)                
                print('从官方模型加载训练！')

                                   
            #创建一个协调器，管理线程
            coord = tf.train.Coordinator()           
            #启动QueueRunner, 此时文件名才开始进队。
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)                      
   
     
            test_acc=0
            test_i=0
            total_cost=0           
            print('开始训练！')
            try:
                #开始一个epoch的训练
                while not coord.should_stop():
                    total_batch = int(train_num/batch_size)
                    #开始一个epoch的训练
                    for i in range(total_batch):
                        
                        train_imgs, train_labs ,train_s= sess.run([train_name_path_batch,train_lable_batch,train_style_batch])  
                        

                        _,loss ,accuracy_value,pred_= sess.run([optimizer,cost,accuracy,pred],feed_dict={input_images:train_imgs,input_style:train_s, input_labels:train_labs,is_training:True})
                        total_cost += loss
                        print(pred_)
                        print(np.argmax(train_labs,axis=1))
                        print("step:",i)
                        print('训练准确率:',accuracy_value)
                        print("loss:",loss)
                    
                        #保存模型
                        if i%100==0:
                            save.save(sess,os.path.join(train_log_dir,train_log_file))
                        
                        if i%10==0:
                            test_i=test_i+1
                            test_imgs, test_labs,test_s = sess.run([test_name_path_batch,test_lable_batch,test_style_batch])          

                            loss ,test_accuracy_value= sess.run([cost,accuracy],feed_dict={input_images:test_imgs,input_style:test_s, input_labels:test_labs,is_training:True})
                            test_acc=test_acc+test_accuracy_value
                            print("step:",i)
                            print('--------------测试准确率:',test_acc/test_i)
                            print('--------------测试准确率:',test_accuracy_value)
            except tf.errors.OutOfRangeError:
                print('Done training')
            finally:
                coord.request_stop()
                coord.join(threads)
                    
                
if __name__ == '__main__':
    main()
