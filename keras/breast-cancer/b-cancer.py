# import all the libraries
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pandas as pd
import keras
from keras import Sequential
from keras.layers import Dense
import sklearn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# load the data
input_data = pd.read_csv('X_data.csv')
label_data = pd.read_csv('Y_datap.csv')

# scale the input data
input_data = preprocessing.scale(input_data)

# split the input data and labels into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    input_data, label_data, test_size=0.2)

# A shallow logistics regression models
model = Sequential()
model.add(Dense(13, input_shape=(30,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001), metrics=['accuracy'])

# early stopping and early stopper
earlystopper = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

# fit model with early stopper with 2000
history = model.fit(X_train, y_train, epochs=2000,
                    validation_split=0.15, verbose=0, callbacks=[earlystopper])
history_dict = history.history

# plot the loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure()
plt.figure()
plt.plot(loss_values, 'b', label='training_loss')
plt.plot(val_loss_values, 'r', label='val training loss')
plt.legend()
plt.xlabel("Epoch")

# plot accuracy over the epoch
accuracy_value = history_dict['acc']
val_accuracy_values = history_dict['val_acc']
plt.plot(val_accuracy_values, '-r', label='val_acc')
plt.plot(accuracy_value, '-b', label='acc')
plt.legend()

# calculate the loss and accuracy of testing data
loss, acc = model.evaluate(X_test, y_test)
print("Test_loss:", loss)
print("Test_accuracy:", acc)

# AUc score of testing data

y_test_pred = model.predict(X_test)
fpr_keras, tpr_keras, thereshold_keras = roc_curve(y_test, y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Testing data AUC', auc_keras)

# ROC curve for testing data
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, labels='Keras (area={:.3f})').format(auc_keras)
plt.xlabel('False positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# AOC score of training
y_train_pred = model.predict_proba(X_train)
fpr_keras, tpr_keras, thereshold_keras = roc_curve(y_train, y_train_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Training data AUC:', auc_keras)

# ROC curve of training
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area={:.3f})'.format(auc_keras))
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# make y_train categorical and assign it to y_train_cat
# sy_train_cat = to_categorical(y_train)
# checkpoints
