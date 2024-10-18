# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/
#数据部分，一直疑惑是早融合还是晚融合，现在MLP中的两层，分别是输入和隐藏，相比我之前的回归少量一层输出（也就是节点是1），
#无法实现完整的MLP回归；同样，cnn也少了一层输出，直到mix train中才有输出，也就是节点为1的层，所以这个既不是早融合，也不是晚融合。就是一个混合版本
# import the necessary packages
import csv

import cv2
import shap
from tensorflow import keras
from pyimagesearch import datasets
from pyimagesearch import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt
import keras
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
# construct the argument parser and parse the arguments
from torchvision.utils import make_grid
import torch.utils.data as data

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")#这里有一个注意点，在OneNote
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "pattern.csv"])#这里是个输入数据的路径
df = datasets.load_house_attributes(inputPath)#其实这个地方还是pd数据，所以用下面的注释也可以
print(type(df), df.shape)#结果<class 'pandas.core.frame.DataFrame'> (1023, 104)
#验证集谱图
testAttrX = df.head(93)
print(type(testAttrX), testAttrX.shape)#结果<class 'pandas.core.frame.DataFrame'> (93, 104)
#训练集谱图
trainAttrX = df.iloc[93:1023, :]
print(type(trainAttrX), trainAttrX.shape)#结果<class 'pandas.core.frame.DataFrame'> (930, 104)
#测试集谱图
pre_end = df.iloc[-43:, :]
print(type(pre_end), pre_end.shape)#结果<class 'pandas.core.frame.DataFrame'> (6, 104)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0#方法1：还用output，方法2，想想办法吧input的数据格式变一下，先看看现在的image是什么格式，（看如何显示）然后变一下
print(type(images), images.shape)#结果<class 'numpy.ndarray'> (1023, 64, 64, 3)
#验证集图像
testImagesX = images[0:93, :]
print(type(testImagesX), testImagesX.shape)#结果<class 'numpy.ndarray'> (93, 64, 64, 3)
#训练集图像
trainImagesX = images[93:1023, :]
print(type(trainImagesX), trainImagesX.shape)#结果<class 'numpy.ndarray'> (930, 64, 64, 3)
#测试集图像
image_end = images[-43:]
print(type(image_end), image_end.shape)#结果<class 'numpy.ndarray'> (6, 64, 64, 3)
#这里可以加一些处理光谱数据df的算法，比如PCA等

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#print("[INFO] processing data...")
#split = train_test_split(data_fon, image_fon, test_size=0.2, random_state=2)
#(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

#定义Y值
#训练集Y
trainY = trainAttrX["E"]
#验证集Y
testY = testAttrX["E"]
#测试集Y
end_Y = pre_end["E"]

'''
# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together.这里是处理光谱数据，最大最小归一，感觉没必要
(trainAttrX, testAttrX) = datasets.process_house_attributes(df,
	trainAttrX, testAttrX)
'''
#单用图像做回归
# create the MLP and CNN models
model = models.create_cnn(16, 16, 3, regress=False)
#model = models.create_mlp_image(64, 64, 3, regress=False)
#可以直接把model改了，用model的回归来做。但是需要考虑优化的问题，以及后续计算的问题
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
#combinedInput = concatenate([mlp.output, cnn.output])#concatenate是合并

# our final FC layer head will have two dense layers, the final one
# being our regression head
#x = Dense(4, activation="relu")(combinedInput)
#x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
#model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-2, decay=1e-2 / 900)#lr就是学习率 learning rate
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)#用法：https://blog.csdn.net/yunfeather/article/details/106461754

# train the model
print("[INFO] training model...")
history = model.fit(
	trainImagesX, trainY,
	validation_data=(testImagesX, testY),
	epochs=100, batch_size=2)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

#绘制loss函数曲线
#acc = history.history['accuracy']#上面没有定义accuracy，在model.compile部分，而上面用平均绝对误差率定义了loss
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss of image_E')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('image_E')
plt.show()

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)#abs求绝对值

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
#print("[INFO] avg. house price: {}, std house price: {}".format(
	#locale.currency(df["density"].mean(), grouping=True),
	#locale.currency(df["density"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
output=csv.writer(open('image_E_val'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], preds.flatten(), testY))# map(lambda), 将python将列表中的数字变为字符才可以输出串，才

'''
#可解释性说明
explain_preds = model.predict(testImagesX)
explainer = shap.GradientExplainer(model, trainImagesX)
shap_values = explainer.shap_values(testImagesX)
shap.image_plot(shap_values, testImagesX)#因为image在前面归一化了，所以导出的shap图像会存在全黑或者全白、全蓝的情况
'''

#数据，使用一个新的csv文件，放在这里，作为测试集使用
#图像

'''
#可解释性说明--废弃，原先想着找几个种类代表，但发现数据加载不出来
transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


explain_dataset_path = ImageFolder(root='./image_gen/testimage', transform=transforms)
explain_dataset = DataLoader(explain_dataset_path, batch_size=12)
'''

print("[INFO] 测试集结果")
explain_preds = model.predict(image_end)

diff_end = explain_preds.flatten() - end_Y
percentDiff_end = (diff_end / end_Y) * 100
absPercentDiff_end = np.abs(percentDiff_end)#abs求绝对值

# compute the mean and standard deviation of the absolute percentage
# difference---测试集的平均相对误差和方差
mean_end = np.mean(absPercentDiff_end)
std_end = np.std(absPercentDiff_end)
print("[INFO] mean_end: {:.2f}%, std_end: {:.2f}%".format(mean_end, std_end))
output=csv.writer(open('Test-image_E_end'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], explain_preds.flatten(), end_Y))

explainer = shap.GradientExplainer(model, trainImagesX)
#print(testAttrX.shape, testImagesX.shape)#输出它们的形状，(19, 25) (19, 64, 64, 3)
shap_values = explainer.shap_values(image_end)
#print(len(shap_values))
#print(len(shap_values[0]))
shap.image_plot(shap_values, image_end)#因为image在前面归一化了，所以导出的shap图像会存在全黑或者全白、全蓝的情况

'''
这段可以用于修改输入数据集
ATR = pd.read_csv("ATR.csv", sep="\t", header=None, encoding='utf-8-sig')#文件得放在mix training.py同一文件夹下
EDX = pd.read_csv("EDX.csv", sep="\t", header=None, encoding='utf-8-sig')
aco = pd.read_csv("acoustic.csv", sep="\t", header=None, encoding='utf-8-sig')
#将三种数据连接起来，这中间可以加一些预处理，如降维、最大最小变化等
df = pd.concat([ATR, EDX, aco], axis=1)
'''
