# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
import csv

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
import pandas as pd
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")#这里有一个注意点，在OneNote
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "2.5HSI-PCA346.csv"])#这里是个输入数据的路径
df = datasets.load_house_attributes(inputPath)#其实这个地方还是pd数据，所以用下面的注释也可以
print(type(df), df.shape)#结果<class 'pandas.core.frame.DataFrame'> (1066, 104)
#验证集谱图
testAttrX = df.head(93)
print(type(testAttrX), testAttrX.shape)#结果<class 'pandas.core.frame.DataFrame'> (93, 1252)
#训练集谱图
trainAttrX = df.iloc[93:1023, :]
print(type(trainAttrX), trainAttrX.shape)#结果<class 'pandas.core.frame.DataFrame'> (930, 104)
#测试集谱图
pre_end = df.iloc[-43:, :]
print(type(pre_end), pre_end.shape)#结果<class 'pandas.core.frame.DataFrame'> (43, 104)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")

#定义Y值
#训练集Y
trainY = trainAttrX["E"]
#验证集Y
testY = testAttrX["E"]
#测试集Y
end_Y = pre_end["E"]

#print(testY)
#光谱取值时需注意，只取前几列的光谱数据，不去后面的label数据
trainAttrX = trainAttrX.iloc[:, :346]#取前连续25列https://blog.csdn.net/weixin_39450145/article/details/115188705
testAttrX = testAttrX.iloc[:, :346]
pre_end = pre_end.iloc[:, :346]

# create the MLP and CNN models
model = models.create_mlp(trainAttrX.shape[1], regress=False)#x_train.shape 是一个用于获取数组或矩阵维度的属性。

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3/300)#lr就是学习率 learning rate
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)#用法：https://blog.csdn.net/yunfeather/article/details/106461754

# train the model
print("[INFO] training model...")
history = model.fit(
	trainAttrX, trainY,
	validation_data=(testAttrX, testY),
	epochs=100, batch_size=2)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testAttrX)

#shap.plots.beeswarm(shap_values)
#print(testAttrX.shape, testImagesX.shape)#输出它们的形状，(19, 25) (19, 64, 64, 3)
#print(testAttrX.columns, testAttrX.index, testAttrX.info)#testAttrX.columns,列索引；testAttrX.index行索引;testAttrX.values,是查看所有值


#绘制loss函数曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss of 2.5HSI-PCA346_E')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('2_5HSI_PCA346_E-2')
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

print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
output=csv.writer(open('val_2.5HSI-PCA346_E-2'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], preds.flatten(), testY))# map(lambda), 将python将列表中的数字变为字符才可以输出串，才

print("[INFO] 测试集结果")
explain_preds = model.predict(pre_end)

#可解释性模型
trainAttrX = np.array(trainAttrX)
explainer = shap.GradientExplainer(model, trainAttrX)

pre_end = np.array(pre_end)
shap_values = explainer.shap_values(pre_end)
#shap.summary_plot(shap_values, pre_end)
data = np.reshape(shap_values, (43, 346))
data1 = pd.DataFrame(np.array(data))
data1.to_csv('SHAP_2.5HSI-PCA346-E-2.csv')

#测试集误差计算
diff_end = explain_preds.flatten() - end_Y
percentDiff_end = (diff_end / end_Y) * 100
absPercentDiff_end = np.abs(percentDiff_end)#abs求绝对值

# compute the mean and standard deviation of the absolute percentage
# difference---测试集的平均相对误差和方差
mean_end = np.mean(absPercentDiff_end)
std_end = np.std(absPercentDiff_end)
print("[INFO] mean_end: {:.2f}%, std_end: {:.2f}%".format(mean_end, std_end))
output=csv.writer(open('test_2.5HSI-PCA346_E-2'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], explain_preds.flatten(), end_Y))

