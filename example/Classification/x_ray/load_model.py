import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers
import numpy as np
# 模型加载

model = tf.keras.models.load_model('./save/save_models/1660ti_cnn.h5')
       
image_value_5 = tf.io.read_file('./pic/n1.jpeg')
image_value_2 = tf.io.read_file('./pic/n2.jpeg')
image_value_0 = tf.io.read_file('./pic/p1.jpeg')

#解码为tensor
image_value_5 = tf.io.decode_jpeg(image_value_5,channels = 1)
image_value_2 = tf.io.decode_jpeg(image_value_2,channels = 1)
image_value_0 = tf.io.decode_jpeg(image_value_0,channels = 1)

image_value_5 = tf.image.resize(image_value_5, (128,128))#改变像素值为128*128
image_value_2 = tf.image.resize(image_value_2, (128,128))#改变像素值为128*128
image_value_0 = tf.image.resize(image_value_0, (128,128))#改变像素值为128*128

#tensor转array
image_value_5 = image_value_5.numpy()
image_value_2 = image_value_2.numpy()
image_value_0 = image_value_0.numpy()

#转为三维数组
image_value_5 = image_value_5.reshape(-1,128,128,1)
image_value_2 = image_value_2.reshape(-1,128,128,1)
image_value_0 = image_value_0.reshape(-1,128,128,1)

#输入模型进行预测
predict_value_5 = model.predict(image_value_5,batch_size = None)
predict_value_2 = model.predict(image_value_2,batch_size = None)
predict_value_0 = model.predict(image_value_0,batch_size = None)

print("")

if np.argmax(predict_value_5) == 1:
    value_5 = '肺炎'
else :
    value_5 = '正常'
if np.argmax(predict_value_2) == 1:
    value_2 = '肺炎'
else :
    value_2 = '正常'
if np.argmax(predict_value_0) == 1:
    value_0 = '肺炎'
else :
    value_0 = '正常'
print("神经网络信息(ACC:76.67%):")
model.summary()
print("导入X光胸片标签： 正常 正常 肺炎")
print("X光胸片已预测完成，对三张X光胸片预测值分别为: ")
print("",value_5,value_2,value_0)
print("武汉 加油！")
print("-xdd_core 正月初一 25/1/2020")

















