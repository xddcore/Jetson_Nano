import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Create a dictionary describing the features.
image_feature_description = {
    "label": tf.io.FixedLenFeature([], tf.int64),
    "img_raw": tf.io.FixedLenFeature([], tf.string),
}
#解析一条example
def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)
#解析n条example  
def read_and_decode(filename): # 读入tfrecords
    i = 0
    img_1 ,label_1 = np.array([]), np.array([])
    img, label = np.array([]), np.array([])
    parsed_image_dataset = tf.data.TFRecordDataset(filename)
    #解析所有example
    parsed_image_dataset = parsed_image_dataset.map (_parse_image_function)
    #取出数据
    for item in parsed_image_dataset:
        #转为向量
        img_1 = np.frombuffer(item['img_raw'].numpy(),dtype=np.uint8).flatten()
        #拼接向量
        img = np.append(img,img_1)
        #转为向量
        label_1 = np.frombuffer(item['label'].numpy(),dtype=np.uint8).flatten()
        #拼接向量
        #获得标签值：0：正常 1：肺炎
        #print((label_1[np.argmax(label_1)]))
        label_1 =np.array([label_1[np.argmax(label_1)]])
        label = np.append(label,label_1)
        i = i + 1
        print("已从TFRecord加载"+str(i)+"张图片...")
    img = img.reshape(-1,128,128,1)
    print(img.shape)
    print(label.shape)
    return img, label

x_train,y_train = read_and_decode("./data/oct_train.tfrecords")
x_test,y_test = read_and_decode("./data/oct_test.tfrecords")
x_valid,y_valid = read_and_decode("./data/oct_val.tfrecords")


# 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：
                          
model = models.Sequential([
                           tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(128,128,1)),
                           tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                           tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
                           tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                           tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
                           tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                           tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(128, activation=tf.nn.relu),
                           layers.Dropout(0.4),
                           tf.keras.layers.Dense(2,  activation=tf.nn.softmax)
])
#model = models.Sequential([layers.Flatten(input_shape=(28, 28)),
#                           layers.Dense(128, activation='relu'),
#                           layers.Dense(128, activation='relu'),
#                           layers.Dense(128, activation='relu'),
#                           layers.Dropout(0.5),
#                           layers.Dense(10, activation='softmax')
#])
# 编译模型
model.compile(optimizer=optimizers.SGD(lr=1e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 打印网络参数量
model.summary()

print(len(model.layers))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_valid.shape)
print(y_valid.shape)
# 训练模型
history = model.fit(x_train, y_train, epochs=500, batch_size = 16,
                  validation_data=(x_test,y_test))
                  
# 验证模型：
model.evaluate(x_valid,  y_valid, verbose=1)


history_dict = history.history         # history对象有一个history成员，它是一个字典，包含训练过程中的所有数据。
print(history_dict)

# 保存模型权重和偏置
model.save_weights('./save/1660ti_cnn/save_weights3/')

#保存完整模型(含网络)
model.save('./save/save_models/1660ti_cnn.h5')



# 绘制loss曲线
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')         # bo代表蓝色圆点
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')    # bo代表蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制acc曲线
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'ro', label='Training acc')           # bo代表蓝色圆点
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')      # bo代表蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()