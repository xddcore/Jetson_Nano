import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

i = 0
j = 0
Show_Number = 0

(train_images,train_labels) ,(test_images,_labels)=keras.datasets.mnist.load_data("./data-mnist")

print("Training data size:",train_images.shape) 

print("Testing data size:",test_images.shape) 

print("Example train data [0]:",train_images[0]) 
print("Example train data shape:",train_images.shape) 

print("Example train data label [0]:",train_labels[0]) 
print("Example train data shape:",train_labels.shape)


fig , ax = plt.subplots(
	nrows = 2,
	ncols = 5,
	sharex = True,
	sharey = True)

fig , bx = plt.subplots(
	nrows = 2,
	ncols = 5,
	sharex = True,
	sharey = True)

ax = ax.flatten()
bx = bx.flatten()

while(i < 10): #show 10 [Show_Number]
	j = j + 1
	if(train_labels[j] == Show_Number):
		img = train_images[j].reshape(28,28)
		ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
		i = i+1
i = 0
j = 0
for k in range(10): #show 0-9 number
		i = 0
		while(i != 1):
			j = j + 1
			if(train_labels[j] == k):
				i = 1
				img = train_images[j].reshape(28,28)
				bx[k].imshow(img, cmap = 'Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
bx[0].set_xticks([])
bx[0].set_yticks([])
plt.tight_layout()
plt.show()


