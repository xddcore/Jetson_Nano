from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

i = 0
j = 0
Show_Number = 0

mnist = input_data.read_data_sets("./data-mnist")

print("Training data size:",mnist.train.num_examples) 

print("Testing data size:",mnist.test.num_examples) 

print("Example train data [0]:",mnist.train.images[0]) 
print("Example train data shape:",mnist.train.images.shape) 

print("Example train data label [0]:",mnist.train.labels[0]) 
print("Example train data shape:",mnist.train.labels.shape)


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
while(i < 10):
	j = j + 1
	if(mnist.train.labels[j] == Show_Number):
		img = mnist.train.images[j].reshape(28,28)
		ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
		i = i+1
i = 0
j = 0
for k in range(10):
		i = 0
		while(i != 1):
			j = j + 1
			if(mnist.train.labels[j] == k):
				i = 1
				img = mnist.train.images[j].reshape(28,28)
				bx[k].imshow(img, cmap = 'Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
bx[0].set_xticks([])
bx[0].set_yticks([])
plt.tight_layout()
plt.show()


