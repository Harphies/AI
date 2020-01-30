# import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
sns.set()

# read the data
data = pd.read_csv(
    "https://github.com/Shreyas3108/Weather/raw/master/weather.csv")
# chech the head of the data
data.head()

# check the columns of the data
data.columns

# check the shape of the data
data.shape

# check the null datas and sum it
data.isnull().sum()

# drop null datas
data = data.dropna()

# check again if all null has been removed
data.isnull().sum()

# check the attributes of the data by describing it
data.describe()

# check the columns of the remaining data
data.columns

# count the value of RainToday Feature
data.RainToday.value_counts()

# plot of a graph
graph = sns.countplot(x='RainToday', data=data)

# distribution plot of minTemp
graph = sns.distplot(data['MinTemp'], bins=40)

# distribution plot of rainfall
graph = sns.distplot(data['Rainfall'])

# check the columns again
data.columns

# data conversions
data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == "Yes" else 0)

train = data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
              'WindDir3pm', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm', 'RainToday', 'RISK_MM']]

data['RainTomorrow'] = data['RainTomorrow'].apply(
    lambda x: 1 if x == "Yes" else 0)

label = data['RainTomorrow']

# check the data type of train data
train.dtypes

# get dummy of the input data
train = pd.get_dummies(train, columns=['WindGustDir', 'WindDir3pm'])

# check the data type pf train data after apply dummy
train.dtypes

# split the data to train and test
x_train, y_train, x_test, y_test = train_test_split(
    train, label, test_size=0.6)

# Instantiate the Logisticregression
model1 = LogisticRegression()

# train with LogReg
model1.fit(x_train, x_test)

# check the score
accuracy_score(y_test, model1.predict(y_train))*100

# confusion matrix
confusion_matrix(y_test, model1.predict(y_train))

# Instantiate Decision tree
model2 = DecisionTreeClassifier()

# train with decisiontree
model2.fit(x_train, x_test)

# check the decision tree accuracy score
accuracy_score(y_test, model2.predict(y_train))*100

# Instantiate Random ForestClassifier
model3 = RandomForestClassifier()

# train with RandomForestClassifier
model3.fit(x_train, x_test)

# check randomForest score
accuracy_score(y_test, model3.predict(y_train))*100
