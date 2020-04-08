# import the libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# import the datasets
fashion_mnist = keras.datasets.fashion_mnist

# split the datas
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
# check the data
print(train_images[0])
# class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               '	Coat', 'Sandal', 'Shirt', 'Sneaker', '	Bag', 'Ankle boot']

# print the shape of the data
print(train_images.shape)
# get the lenght of the training images
print(len(train_images))
# get the lenght and shape of test Images
print(test_images.shape)
# get the lenght of test images
print(len(test_images))
# get the dimension of labels
print(train_labels.shape)
print(test_labels.shape)
# get their lenghts
print(train_images)
print(test_images)

""""
# plot the images
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



# plot the first 10images
plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
# normalize or scale the images before feed into neural network
train_images = train_images/255.0
test_images = test_images/255.0

# define the model
model = keras.Sequential()
# Add the input layer
# Flatten layer convert 2D to 1D by flatten the data
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# Hidden layer
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# output layer
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# compile the model with the optimizer, loss, matrics
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# fit the model
model.fit(train_images, train_labels, epochs=5)

# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test:Accuracy{test_acc}")

# make predicition
prediction = model.predict(test_images)

# print the test_image[0]
print(prediction[0])

# print the mag of what the network thinks it's
print(np.argmax(prediction[0]))

# print the actual image_label 0 to see if the network is right
print(test_labels[0])
