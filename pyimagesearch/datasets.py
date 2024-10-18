# import the necessary packages
from opt_einsum.backends import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	df = pd.read_csv(inputPath, header=0, encoding='utf-8-sig')#sep="\t",---header的默认值为0，即将数据第一行作为列名；header=None时，认为数据中没有列名，pandas自动设置了列名；sep：读取文件时指定的分隔符，默认为逗号。注意："csv文件的分隔符" 和 "我们读取csv文件时指定的分隔符" 一定要一致。\t为制表符

	#这一段可以加一些数据预处理的方式

	# return the data frame
	return df
'''
def process_house_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["bedrooms", "bathrooms", "area"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()#红外不考虑最大最小归一化
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoing, all output features are now in the range [0, 1])--这里这部分编码，可以用于分类的时候编码；不过回归不需要编码
	#zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	#trainCategorical = zipBinarizer.transform(train["zipcode"])
	#testCategorical = zipBinarizer.transform(test["zipcode"])

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	#trainX = np.hstack([trainCategorical, trainContinuous])
	#testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)
'''
'''
def load_house_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []


	#imgList = os.listdir(inputPath)#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。https://www.runoob.com/python/os-listdir.html
	imgList = os.path.sep.join([inputPath])
	imgList.sort(key=lambda x: int(x.split("_")[0]))

	for i in df.index.values:#将数据及与图像对应，df.index.values #返回序列对象
		filename = imgList[i]
		image = cv2.imread(inputPath + "/" + filename)  # 根据图片名读入图片

		basePath = os.path.sep.join([inputPath])
		#housePaths = sorted(list(glob.glob(inputPath)))

		inputImages = []

		image = cv2.imread(inputPath)

		image = cv2.resize(image, (64, 64))#调用cv2.resize（）通过插值的方式来改变图像的尺寸
		images.append(image)

	#images.append(inputImages)
	# return our set of images
	return np.array(images)
'''
def load_house_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []

	# loop over the indexes of the houses
	for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
		basePath = os.path.sep.join([inputPath, "{}_*".format(i)])#"{}_*".format(i + 1)，表示查找_的数量，并将四张图片排序
		housePaths = sorted(list(glob.glob(basePath)))#按名称排序

		# initialize our list of input images along with the output image
		# after *combining* the four input images

		outputImage = np.zeros((16, 16, 3), dtype="uint8")

		# loop over the input house paths
		for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
			image = cv2.imread(housePath)
			image = cv2.resize(image, (16, 16))#这里32换大一些，resize是缩放
			outputImage[0:16, :, :] = image
			#以下3句用来确定每张图片是否正确
			#cv2.imshow("src", outputImage)
			# 等待显示
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()#每张图片出来后要按任意键盘键切换下一张图片
		'''
			#print(image, len(image), type(image))
			#inputImages.append(image)
			#print(inputImages, len(inputImages))
			#print(inputImages.shape)
		#outputImage[0:64] = image
		# add the tiled image to our set of images the network will be
		# trained on
		'''
		images.append(outputImage)

	#	np.mat(images, dtype="uint8")#把列表转换为矩阵
	#print(type(images))

	# return our set of images
	return np.array(images)


