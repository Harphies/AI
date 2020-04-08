# import all the libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing import image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Instantiate the classifier from Sequential
classifier = Sequential()

# Initializing the first layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

# Add a dense layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# compile the classifier
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train and test data
train_data = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

# train
train = train_data.flow_from_directory(
    'dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test = test_data.flow_from_directory(
    'dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# classifier
classifier.fit_generator(train, steps_per_epoch=8000,
                         epochs=10, validation_data=test, validation_steps=2000)

# import sample image to test
imagetest = image.load_img('image1.jpg', target_size=(64, 64))

# call the image to see if imported
imagetest

# adding the image as an array N:D it's added as 3D array by default
imagetest = image.img_to_array(imagetest)

# modify to 1D array
imagetest = np.expand_dims(imagetest, axis=0)

# make a predicition
train.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# results
results = classifier.predict(imagetest)
# prediction
prediction
