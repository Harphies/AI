# import the packages
import sklearn
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import joblib

# load the data
data = pd.read_csv("dataset.csv")

# split the data into output and Input
X = data.drop(["Label"], axis=1)
y = data["Label"]

# split into train and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# build the model
model = RandomForestClassifier(n_estimators=100, max_depth=5)

# train the model
model.fit(X_train, y_train)

# save the model
joblib.dump(model, "malaria_05")

# make prediction
prediction = model.predict(X_test)

# print the prediction score
print(metrics.classification_report(prediction, y_test))
print(model.score(X_test, y_test))
