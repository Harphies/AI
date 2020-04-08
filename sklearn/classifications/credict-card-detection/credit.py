import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# load the data
data = pd.read_csv("creditcard.csv", sep=',')
data.head()

# get the data Info
data.info()


# perfom some EDA
data.isnull().values.any()

# count all the labels
count_classes = pd.value_counts(data['Class'], sort=True)

# plot the count class
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

# classify the fraud and Normal dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print(fraud.shape, normal.shape)

#  More information about the fraud transaction
fraud.Amount.describe()

# More information about normal cards
normal.Amount.describe()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle("Amount per transaction by class")
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins=bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transaction')
plt.xlim(0, 20000)
plt.yscale('log')
plt.show()

# let's check if fraud occurs over a specific time interval
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# let's take some sample data
data1 = data.sample(frac=0.1, random_state=1)
data1.shape

# original data
data.shape

# detrmine the number of fraud and valid transaction in the dataset
Fraud = data1[data1['Class'] == 1]
Valid = data1[data1['Class'] == 0]


outlier_fraction = len(Fraud)/float(len(Valid))

print(outlier_fraction)

print("Fraud Cases: {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

# Correlation

# get correlation of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")

# create dependent and Independent features
columns = data1.columns.tolist()

# filter the unwanted data
columns = [c for c in columns if c not in ["Class"]]
# store the Variable we are predicting
targets = "Class"
# Define a rnadom state
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[targets]
X_outlier = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# print the shape of X and Y
print(X.shape)
print(Y.shape)

# Build the Model
# Isolation Forest Algorithm, Local Outlier Factor


# Define the Outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X)),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine": OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.05,
                                          max_iter=-1, random_state=state)
}

type(classifiers)

# Number of outlier
n_outliers = len(Fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data tag the outlier
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    # reshape the prediction
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run classification metrics
    # Run Classification Metrics
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y, y_pred))
    print("Classification Report :")
    print(classification_report(Y, y_pred))
