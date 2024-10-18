# import the necessary packages

from tensorflow import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(24, input_dim=dim, activation="elu"))#units ：代表该层的输出维度或神经元个数, units解释为神经元个数为了方便计算参数量，解释为输出维度为了方便计算维度
	model.add(Dense(20, activation="elu"))
	#model.add(Dense(24, activation="elu"))
	model.add(Dense(16, activation="elu"))
	#model.add(Dense(8, activation="elu"))
	model.add(Dense(8, activation="elu"))
	# check to see if the regression node should be added
	#if regress:#最上面和混合模型都，有关于regress=False的设定，到时候看看改不改
	model.add(Dense(1, activation="linear"))

	# return our model
	return model

def create_mlp2(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="elu"))
	model.add(Dense(4, activation="sigmoid"))
	#model.add(Dense(2, activation="elu"))
	# check to see if the regression node should be added
	if regress:#最上面和混合模型都，有关于regress=False的设定，到时候看看改不改
		model.add(Dense(1, activation="linear"))

	# return our model
	return model

def create_cnn(width, height, depth, filters=(8, 16, 16), regress=False):#这里是一个三维的filter
#关于filter层数，所以后一层卷积层需要增加feature map的数量，才能更充分的提取出前一层的特征，所有后一层的filter比前一层一般是成倍增加
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):#"enumerate"函数在字典中的意思是“枚举”或“列举”。当使用"enumerate"时，通常会在for循环中得到计数，同时获得索引和值。例如，如果你有一个列表并想在循环中同时打印元素的索引和值，你可以使用"enumerate"，
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)#f为filter，（3,3）为步长，padding=same表示【输出层尺寸 = 输入层尺寸】，边缘外自动补0，遍历相乘。strides=1,是我加上的，但是mixed training效果变差了
		x = Activation("elu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(1, 1))(x)#如果是32*32的图像，pool size用（1,1）。图像是64*64，，pool size用（2,2）。图片的相邻像素具有相似的值，因此卷基层中很多信息是冗余的。通过池化来减少这个影响，包含 max, min or average，下图为基于 2x2 的 Max Pooling：
		x = Conv2D(f, (3, 3), padding="same")(x)#f为filter，（3,3）为步长，padding=same表示【输出层尺寸 = 输入层尺寸】，边缘外自动补0，遍历相乘。strides=1,是我加上的，但是mixed training效果变差了
		x = Activation("elu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(1, 1))(x)#, dim_ordering="th"

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(24)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)#batch_normalization一般是用在进入网络之前，它的作用是可以将每层网络的输入的数据分布变成正态分布，有利于网络的稳定性，加快收敛。没有什么超参数可以设置
	x = Dropout(0.5)(x)#也是为了防止过拟合，但是有帖子说，dropput更适合于大数据集，小数据集没有太大必要。240423，暂未发现明显影响，先用

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP，匹配节点的数量不是必须的，但有助于平衡分支
	x = Dense(12)(x)
	x = Activation("relu")(x)
	x = Dense(10)(x)
	x = Activation("elu")(x)
	x = Dense(6)(x)
	x = Activation("elu")(x)
	x = Dense(4)(x)
	x = Activation("elu")(x)


	# check to see if the regression node should be added
	#if regress:
	x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

#这是一段实验，废弃240721
def create_mlp_image(width, height, depth, filters=(16, 32, 64), regress=False):#这里是一个三维的filter
	#这样就是没有卷积，直接将图像输入mlp，其实是少了图像卷积的预处理。其他没影响。
	# 关于filter层数，所以后一层卷积层需要增加feature map的数量，才能更充分的提取出前一层的特征，所有后一层的filter比前一层一般是成倍增加
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	x = inputs
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(
		x)  # batch_normalization一般是用在进入网络之前，它的作用是可以将每层网络的输入的数据分布变成正态分布，有利于网络的稳定性，加快收敛。没有什么超参数可以设置
	x = Dropout(0.5)(x)  # 也是为了防止过拟合，但是有帖子说，dropput更适合于大数据集，小数据集没有太大必要。240423，暂未发现明显影响，先用

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP，匹配节点的数量不是必须的，但有助于平衡分支
	x = Dense(12)(x)
	x = Activation("elu")(x)
	x = Dense(10)(x)
	x = Activation("relu")(x)
	x = Dense(10)(x)
	x = Activation("elu")(x)
	x = Dense(6)(x)
	x = Activation("elu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

'''
#240711-试验后失败--用来添加未知样本
def predict_data(path, transforms=transforms):
	transforms = transforms.Compose([
		transforms.Resize(64),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])
	predict_path = ImageFolder(path, transform=transforms)
	predict = DataLoader(predict_path, batch_size=6)

	return predict
'''