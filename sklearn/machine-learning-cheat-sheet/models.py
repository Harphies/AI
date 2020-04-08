# sklearn
import sklearn
from sklearn.metrics import roc_curve, mean_squared_error, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.external import joblib
from sklearn.model_selection import GridSearchCV

# pandas
import pandas as pd


# Matplotlib
import matplotlib.pyplot as plt


# seaborn
import seaborn as sns


# keras
import keras
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# Load a data and scale it
input_data = pd.read_csv('')

'''

# scale the dataset
scaler = Scaler()
X = scaler.fit_transform(Xfeatures)

standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(
    dataset[columns_to_scale])

'''

'''
# split data

# split the data into input and output
y = dataset['target']
X = dataset.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, Ylabel, test_size=0.2, random_state=42)

'''
'''

# Calculate loss and accuracy of testing data
loss, acc = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", acc)

'''

'''
Moddels
# Building the Model with Logreg anf KNN
logreg = LogisticRegression()
# train the model
logreg.fit(x_train, y_train)
# check acuracy
logreg.score(x_test, y_test)

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=12)
score = cross_val_score(knn_classifier, X, y, cv=10)
score.mean()

# random Forest
randomforest_classifier = RandomForestClassifier(n_estimators=10)
score = cross_val_score(randomforest_classifier, X, y, cv=10)
score.mean()

# instantiate SVC
svm_model = SVC()
# train with SVC
svm_model.fit(train_x, test_x)
# make prediction with SVC
predict3 = svm_model.predict(train_y)

# Increase the acuracy of the model with Gradient boosting
clf = GradientBoostingRegressor(
    n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
# train the model
clf.fit(x_train, y_train)
# check acuracy
clf.score(x_test, y_test)



'''

'''

# make prediction
X_test.values[0]

# Prediction on A Single Sample
logreg.predict(np.array(X_test.values[0]).reshape(1, -1))

'''

'''
knn_scores = []
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier, X, y, cv=10)
    knn_scores.append(score.mean())
'''

'''
# Categorical data conversion to numric 
# note both label encoder and onehot are used for categorical data conversion

# instantiate Label encoder
le = LabelEncoder()

# re-label the categorical features
data.WindGustDir = le.fit_transform(data.WindGustDir)
data.WindDir3pm = le.fit_transform(data.WindDir3pm)
data.WindDir9am = le.fit_transform(data.WindDir9am)

'''

'''
# how to use accuracy score
model.fit(train_x, test_x)
#make a prediction
predict = model.predict(train_y)
# check the accuracy score
accuracy_score(predict, test_y)
# check accuracy score of SVC
accuracy_score(predict3, test_y)
'''

'''
compute mean suare error
pred = logreg.predict()
mse = mean_square_error(label,pred)
rmse = np.sqrt(mse)
'''

'''
using cross validation score by dividing the data into folds
score = cross_val_score(tree_reg, data,labels,scoring="neg_mean_squared_error",cv=10) # cv = corss validation
rmse_score = np.sqrt(scores)


'''

'''
save a model using joblib or pickle
jpblib.dump(my_model,'my_model.pkl')

# load it later
saved_model = joblib.load('my_model.pkl')
'''

'''
Fine Tuning with Grid search


'''
