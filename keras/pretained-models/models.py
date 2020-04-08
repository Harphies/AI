from keras.applications.vgg19 import VGG19
import keras
from keras.models import Model
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# load the model
model = ResNet50(weights='imagenet')


# get the image path
img_path = ''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prediction
prediction = model.prediction(x)

print("Predicted:", decode_predictions(prediction, top=5)[0])

# Extract layers and features
base_model = VGG19(weights='imagenet')
base_model.summary()
