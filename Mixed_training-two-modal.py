# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/
#数据部分，一直疑惑是早融合还是晚融合，现在MLP中的两层，分别是输入和隐藏，相比我之前的回归少量一层输出（也就是节点是1），
#无法实现完整的MLP回归；同样，cnn也少了一层输出，直到mix train中才有输出，也就是节点为1的层，所以这个既不是早融合，也不是晚融合。就是一个混合版本
# import the necessary packages
import csv

import shap as shap
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
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")#这里有一个注意点，在OneNote
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "peakpattern.csv"])#这里是个输入数据的路径
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
#print(images.shape)
images = images / 255#数据归一化。方法1：还用output，方法2，想想办法吧input的数据格式变一下，先看看现在的image是什么格式，（看如何显示）然后变一下
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
print("[INFO] processing data...")

#定义Y值
#训练集Y
trainY = trainAttrX["C"]
#验证集Y
testY = testAttrX["C"]
#测试集Y
end_Y = pre_end["C"]

#光谱取值时需注意，只取前几列的光谱数据，不去后面的label数据
trainAttrX = trainAttrX.iloc[:, :14]#取前连续25列https://blog.csdn.net/weixin_39450145/article/details/115188705
testAttrX = testAttrX.iloc[:, :14]
pre_end = pre_end.iloc[:, :14]

'''
# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together.这里是处理光谱数据，最大最小归一，感觉没必要
(trainAttrX, testAttrX) = datasets.process_house_attributes(df,
	trainAttrX, testAttrX)
'''

# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)#trainAttrX.shape[1],查看列数
cnn = models.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])#concatenate是合并

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="linear")(combinedInput)
#x = Dense(3, activation="relu")(x)
x = Dense(2, activation="elu")(x)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-2, decay=1e-2 / 550)#lr就是学习率 learning rate
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)#用法：https://blog.csdn.net/yunfeather/article/details/106461754

# train the model
print("[INFO] training model...")
history = model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=100, batch_size=2)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

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
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["C"].mean(), grouping=True),
	locale.currency(df["C"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
output=csv.writer(open('Val-Deep-fusion-HSI-peak+RGBimage-0719_E-2'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], preds.flatten(), testY))# map(lambda), 将python将列表中的数字变为字符才可以输出串，才

#绘制loss函数曲线
#acc = history.history['accuracy']#上面没有定义accuracy，在model.compile部分，而上面用平均绝对误差率定义了loss
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss of Deep-fusion-HSI-peak+RGB image_E')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss-Deep-fusion-HSI-peak+RGBimage-0719_E-2')
plt.show()

#未知样品测试
print("[INFO] 测试集结果")
explain_preds = model.predict([pre_end, image_end])

diff_end = explain_preds.flatten() - end_Y
percentDiff_end = (diff_end / end_Y) * 100
absPercentDiff_end = np.abs(percentDiff_end)#abs求绝对值

# compute the mean and standard deviation of the absolute percentage
# difference---测试集的平均相对误差和方差
mean_end = np.mean(absPercentDiff_end)
std_end = np.std(absPercentDiff_end)
print("[INFO] mean_end: {:.2f}%, std_end: {:.2f}%".format(mean_end, std_end))
output=csv.writer(open('Test-Deep-fusion-HSI-peak+RGBimage-0719_E-2'+'.csv','a+',newline=''),dialect='excel')
output.writerows(map(lambda x, y: [x, y], explain_preds.flatten(), end_Y))

# 保存模型，240509适合在进行迁移学习时。但是每一种输入下，预测不同特性时，都需要保存一个模型，对于我来说还挺麻烦
# model.save('ConV_DT.h5')
#这一步是整个融合过程中较为重要的步骤，只有较为准确的CNN网络才能提取准确有用的特征，一般采用预训练+微调的模式来训练CNN，为了演示方便，我们采用Fashion MNIST数据集来为我们的图像，iris数据集作为我们的数值型数据，
# 来做演示。另外由于Fashion MNIST类别有10中，而iris只有3种，我们之纳入irsi的数据来“假定为我们图像配套的数值型数据”，新建一个py文件，具体代码如下：
#原文链接：https://blog.csdn.net/JaysonWong/article/details/126628707

''' 这里因为红外数据与图像数据的shape（形状）不一致，所以无法进行
# DeepExplainer to explain predictions of the model
#backgroundm = trainAttrX[np.random.choice(trainAttrX.shape[0],100, replace=True)]#https://blog.csdn.net/ImwaterP/article/details/96282230
#explainer = shap.DeepExplainer(model.predict, backgroundm, session=None, learning_phase_flags=None)
#backgroundn = trainImagesX[np.random.choice(trainImagesX.shape[0],100, replace=True)]
实验2
explainer = shap.GradientExplainer(model, [trainAttrX, trainImagesX])
#testAttrX.columns = testAttrX.reset_index()
#testAttrX = np.array(testAttrX)
print(trainAttrX, trainImagesX)
#print(testAttrX)
#print(testAttrX.shape, testImagesX.shape)#输出它们的形状，(19, 25) (19, 64, 64, 3)
shap_values = explainer.shap_values([testAttrX, testImagesX])
print(len(shap_values))
print(len(shap_values[0]))
#shap.image_plot([shap_values[i][0] for i in range(10)], x_test[:3])
'''



'''
这段可以用于修改输入数据集
ATR = pd.read_csv("ATR.csv", sep="\t", header=None, encoding='utf-8-sig')#文件得放在mix training.py同一文件夹下
EDX = pd.read_csv("EDX.csv", sep="\t", header=None, encoding='utf-8-sig')
aco = pd.read_csv("acoustic.csv", sep="\t", header=None, encoding='utf-8-sig')
#将三种数据连接起来，这中间可以加一些预处理，如降维、最大最小变化等
df = pd.concat([ATR, EDX, aco], axis=1)
'''
