# -*- coding: utf-8 -*- 
from __future__ import print_function
import numpy as np 
import os
import pickle as cPickle
import math
import shutil
import matplotlib
import cv2
import csv
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import pandas as pd
####################################################
###calculating the representativity for all artist##
###You can also calculate the specific artist by changing the ARTIST PATH##
####################################################




csvFile = open("score208.csv", "r")
reader = csv.reader(csvFile)
score1=[]
for item in reader:
	if reader.line_num == 1:
		continue
	score1.append("../Images/"+item[1]+"/"+item[2])
csvFile.close()
def plot_embedding(data,  image,title):
	size = 512	
	fig = plt.figure()
	ax = plt.subplot(111)
	for i in range(data.shape[0]):
		#print(image[i])
		plt.text(data[i, 0], data[i, 1],str(i),color=plt.cm.Set1(1),
			fontdict={'weight': 'bold', 'size': 1})
	if hasattr(offsetbox, 'AnnotationBbox'):
	    # only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1.]])  # just something big
		for i in range(data.shape[0]):
            
			# dist = np.sum((data[i] - shown_images) ** 2, 1)
			# if np.min(dist) < 2e-5:
	  #               # don't show points that are too close
			# 	continue
			# shown_images = np.r_[shown_images, [data[i]]]
			#print("vincent-van-gogh/"+image[i])
			img = cv2.imread(path+"/"+image[i])[:,:, ::-1]
			if len(img.shape)==3:
				w,h ,c= img.shape
			

				if w >= h:
					ratio = float(h)/float(w)
					resize_factor = (int(size/ratio), size)
					img_resize = cv2.resize(img, resize_factor)
				else:
					ratio = float(w)/float(h)
					resize_factor = (size, int(size/ratio))
					img_resize = cv2.resize(img, resize_factor)
				w,h,c= np.shape(img_resize)
				crop_w = int((w-size) * 0.5)
				crop_h = int((h-size) * 0.5)
				
				img = img_resize[crop_w:crop_w+size,crop_h:crop_h+size,:]
				#plt.imshow(img)
				# plt.savefig("{}.png".format(i))
				# if i > 10:
				# 	break

				#print(img.shape)
				if path+"/"+image[i] in score1:
					print(path+"/"+image[i])
					imagebox = offsetbox.AnnotationBbox(
						offsetbox.OffsetImage(img, zoom=0.6,cmap=None),
						data[i],frameon=True,bboxprops={"color":"r"}, pad=0.01)
				else:
				    imagebox = offsetbox.AnnotationBbox(
				    	offsetbox.OffsetImage(img, zoom=0.6,cmap=None),
				    	data[i],frameon=False, pad=0.0)
				#print(i, type(imagebox))
				ax.add_artist(imagebox)

	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)

def azimuthAngle( x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    #return (angle * 180 / math.pi)
    return angle

#读数据
read_file=open("kpl/reference_point30_all.kpl",'rb')
feature_nonsocre=cPickle.load(read_file)              #没有代表性的画作特征
image_name_nonscore=cPickle.load(read_file)           #没有代表性的画作名
image_lable_nonscore=cPickle.load(read_file)          #没有代表性的画作类别
k_n_feature_collection=cPickle.load(read_file)        #没有代表性的画作周围k个有代表性的画作特征
k_n_image_collection=cPickle.load(read_file)          #没有代表性的画作周围k个有代表性的画作名
k_n_dis_collection=cPickle.load(read_file)            #没有代表性的画作周围k个有代表性的画作与该画作的距离
k_n_lab_collection=cPickle.load(read_file)            #没有代表性的画作周围k个有代表性的画作类别
ar=[]
rs=[]
artists=[]
csvFile = open("100_2.csv", "r")
#csvFile = open("list500_5.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    if reader.line_num == 1:
        continue
    artists.append(item[1])
csvFile.close()
artists=list(set(artists))

for artist in artists:
	print("#########",artist,"################")
	if not tf.gfile.Exists("sorted_image/"+artist):
		tf.gfile.MakeDirs("sorted_image/"+artist)
	# shutil.rmtree("negative_m2/")
	# os.mkdir("negative_m2/")
	# shutil.rmtree("positive_m2/")
	# os.mkdir("positive_m2/")
	csvFile = open("all_year.csv", "r")
	reader = csv.reader(csvFile)
	painting_year = {}
	year=[]
	for item in reader:
		if reader.line_num == 1:
			continue
		#print(item[2])
		if item[1]==artist:
			#print("--------")
			painting_year["../Images/"+item[1]+"/"+item[2]]=item[6] 
			if item[6] !="":
				a=int(item[6])
				year.append(a)
	csvFile.close()

	weight=1
	path="../Images/"+artist
	fs = os.listdir(path)
	print("Painting number",len(fs))
	re_=[]
	image_=[]
	for image in fs: 
		if path+"/"+image in image_name_nonscore:
			
			ind = image_name_nonscore.index(path+"/"+image)
			feature = feature_nonsocre[ind]
			lable = image_lable_nonscore[ind]

			features = k_n_feature_collection[ind]

			k=len(features)
			#print(k)
			labs = k_n_lab_collection[ind]
			angles=[]
			for i in range(k):
				angle = azimuthAngle(feature[0],feature[1],features[i][0],features[i][1])
				angles.append(angle)
			sortedAngIndicies = np.argsort(angles)#排序并返回index  down
			sortedAngles=sorted(angles,reverse=False)# down
			
			w_positive=0
			w_negative=0
			w=[]
			#w_sum=0
			#print(image,lable)
			for j in range(k):
				#计算所有的点
				# if j==0:

				# 	a = angles[sortedAngIndicies[1]] - angles[sortedAngIndicies[0]]
				# 	a_ = 2*math.pi - angles[sortedAngIndicies[k-1]] + angles[sortedAngIndicies[1]]
				# elif j==k-1:
				# 	a = 2*math.pi - angles[sortedAngIndicies[k-1]] + angles[sortedAngIndicies[1]]
				# 	a_ = angles[sortedAngIndicies[k-1]] - angles[sortedAngIndicies[k-2]]
				# elif 0<j<k-1:
				# 	a = angles[sortedAngIndicies[j+1]] - angles[sortedAngIndicies[j]]
				# 	a_ = angles[sortedAngIndicies[j]] - angles[sortedAngIndicies[j-1]]
				# w_=abs((math.tan(a/2)+math.tan(a_/2))/np.linalg.norm(features[sortedAngIndicies[j]] - feature, ord=1))
				# w.append(w_)
				##################################################
				#不计算边缘点
				if j==0 or j==k-1:
					w.append(0)    
				if 0<j<k-1:
					a = angles[sortedAngIndicies[j+1]] - angles[sortedAngIndicies[j]]
					a_ = angles[sortedAngIndicies[j]] - angles[sortedAngIndicies[j-1]]
					# w_=abs((math.tan(a/2)+math.tan(a_/2))/np.linalg.norm(features[sortedAngIndicies[j]] 
					# 	- feature, ord=1))
					w_=(math.tan(a/2)+math.tan(a_/2))/np.linalg.norm(features[sortedAngIndicies[j]] 
						- feature, ord=1)
					w.append(w_)
				#####################################################
				#w_sum=w_sum+abs((math.tan(a/2)+math.tan(a_/2))/np.linalg.norm(features[sortedAngIndicies[j]] - feature, ord=1))
				# if image == "the-starry-night-1888-2.jpg":
				# 	print(math.tan(a/2),math.tan(a_/2),w_,labs[sortedAngIndicies[j]])

			for j in range(k):
				#print(type(labs[sortedAngIndicies[j]]),type(lable))
				if int(labs[sortedAngIndicies[j]] ) == lable:
					#print("----------1",w[j])
					w_positive = w_positive + np.abs(w[j])
					#w_sum = w_sum +weight*w[j]
				else :
					#print("----------2",w[j])
					w_negative = w_negative + np.abs(w[j])
					#w_sum = w_sum +w[j]
			try:
			
			
				weight=int(painting_year[path+"/"+image])-min(np.array(year))+1
			except:

				weight=1
				#weight=1
			#print(weight)
			#weight = 1
			re=(weight*w_positive-0.01*w_negative)

			re_.append(re)
			image_.append(image)
			
			# if re >0:
			
			# 	#print(image,lable,re)
			# 	shutil.copy(path+"/"+image, "positive_m/"+image)
			# if re <0:
				
			# 	shutil.copy(path+"/"+image, "negative_m/"+image)

	################visual##############
	# print("----",max(abs(np.array(re_))))
	denominator=np.log10(max(abs(np.array(re_))))
	sorting=[]
	for i in range(len(re_)):
		if re_[i]>0:
			sorting.append(np.log10(re_[i])/denominator)
		else:
			sorting.append(np.log10(abs(re_[i]))/denominator)
	#re_=re_/max(abs(np.array(re_)))
	# sorting=re_
	# denominator=max(abs(np.array(re_)))
	sortedReIndicies = np.argsort(-np.array(sorting) )   

	stanordi=[]
	for i in range(40):
		for j in range(50):
			stanordi.append([0.02*j,0.8-0.02*i])
	ordi=[]
	sortimg=[]
	a=0
	k=0
	for i in range(len(image_)):
		# if image_[i] in os.listdir("vincent"):
		print("-----",image_[i],sorting[i])
		# #print(path+"/"+image_[i])
		if path+"/"+image_[sortedReIndicies[i]] in score1:
			#print(1-sorting[sortedReIndicies[i]])
			#shutil.copy(path+"/"+image_[sortedReIndicies[i]], "labeled/{}.png".format(i))

			a=a+(sorting[sortedReIndicies[i]])
			k=k+1
			#print(image_[i],1-sorting[sortedReIndicies[i]])
			#print(image_[sortedReIndicies[i]],sorting[sortedReIndicies[i]])
		if i==762:
			x=sorting[sortedReIndicies[i]]
		ordi.append(stanordi[i])
		sortimg.append(image_[sortedReIndicies[i]])
		shutil.copy(path+"/"+image_[sortedReIndicies[i]], "sorted_image/"+"test"+"/{}.png".format(i))
		#shutil.copy(path+"/"+image_[sortedReIndicies[i]], "sorted_image/"+"test"+"/"+image_[sortedReIndicies[i]])

	ordi=np.array(ordi)
	try:
		print(artist,a/k,k)
		ar.append(artist)
		rs.append(a/k)
	except:
		ar.append(artist)
		rs.append(0)


c=np.array([ar,rs])
c=c.T
name=['artist','re']
test=pd.DataFrame(columns=name,data=c)
test.to_csv("re_30.csv",encoding='utf-8')

	# plot_embedding(ordi,sortimg,'result')  
	# plt.savefig("new_png/"+artist+"30.png", dpi=2880)  



