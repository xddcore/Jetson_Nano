import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers
import numpy as np

# 模型加载

model = tf.keras.models.load_model('./save/1660ti_tf2.1_py3.7/save_models/hdf5/1660ti_1.h5')

image_value_5 = tf.io.read_file('./pic/5.png')
image_value_2 = tf.io.read_file('./pic/2.png')
image_value_0 = tf.io.read_file('./pic/0.png')

#解码为tensor
image_value_5 = tf.io.decode_png(image_value_5,channels = 1)
image_value_2 = tf.io.decode_png(image_value_2,channels = 1)
image_value_0 = tf.io.decode_png(image_value_0,channels = 1)

#tensor转array
image_value_5 = image_value_5.numpy()
image_value_2 = image_value_2.numpy()
image_value_0 = image_value_0.numpy()

#转为三维数组
image_value_5 = image_value_5.reshape(1,28,28)
image_value_2 = image_value_2.reshape(1,28,28)
image_value_0 = image_value_0.reshape(1,28,28)

#输入模型进行预测
predict_value_5 = model.predict(image_value_5,batch_size = None)
predict_value_2 = model.predict(image_value_2,batch_size = None)
predict_value_0 = model.predict(image_value_0,batch_size = None)

print("")
print("预测完成，预测值分别为: ",np.argmax(predict_value_5),np.argmax(predict_value_2),np.argmax(predict_value_0))