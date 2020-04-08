# import all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neigbhors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# import the data and check features
data = pd.read_csv('weather_data.csv')
data.head()

# data size
data.size

# data shape
data.shape

# data columns
data.columns

# plot a bar graph for winddir9am
data.WindDir9am.value_counts().plot(kind='barh')
plt.title('Wind Direction 9am')
plt.show()

# plot wind direction 3pm
data.WindDir3pm.value_counts().plot(kind='barh')
plt.title('Wind direction 3pm')
plt.show()

# convert categorical data to numerical data
data.RainToday = data.RainToday.apply(lambda x: 1 if x == "Yes" else 0)

# convert categotical data to muneric data
data.RainTomorrow = data.RainTomorrow.apply(lambda x: 1 if x == "Yes" else 0)

# check the conversion fro head
data.head()

# plot for the windGustDir
data.WindGustDir.value_counts().plot(kind='barh', color='c')
plt.title('Wind Gust Direction')
plt.show()

# instantiate Label encoder
le = LabelEncoder()

# remove missing values
data = data.dropna()

# check the shape again wether it has removed missed value
data.shape

# re-label the categorical features
data.WindGustDir = le.fit_transform(data.WindGustDir)
data.WindDir3pm = le.fit_transform(data.WindDir3pm)
data.WindDir9am = le.fit_transform(data.WindDir9am)

# check the datas wether ithas been converted
data.describe()

# separate Inputs from outputs
x = data.drop(['Date', 'Location', 'RainTomorrow'], axis=1)

# outputs or labels
y = data.RainTomorrow

train_x, train_y, test_x, test_y = train_test_split(
    x, y, test_size=0.2, random_state=2)

# check the size of input train data
train_x.shape
# check the size of the output train data
train_y.shape

# instantiate Logistics Regression
model = LogisticRegression()

# train the model
model.fit(train_x, test_x)

# make a prediction
predict = model.predict(train_y)

# check the accuracy score
accuracy_score(predict, test_y)

# instantiate SVC
svm_model = SVC()

# train with SVC
svm_model.fit(train_x, test_x)

# make prediction with SVC
predict = svm_model.predict(train_y)

# check accuracy score of SVC
accuracy_score(predict, test_y)

# Instantiate KNN
clf = KNeighborsClassifier()

# train with KNN
clf.fit(train_x, test_x)

# make predicition with KNNs
predict = clf.predict(train_y)

# check accuracy score
accuracy_score(predict, test_y)

# Instantiate RandomForest
Rf = RandomForestClassifier(max_depth=4)

# train with RandomForestCassifier
Rf.fit(train_x, test_x)

# make prediction with RandomForest
predict = Rf.predict(train_y)

# check the accuracy score
accuracy_score(predict, test_y)
