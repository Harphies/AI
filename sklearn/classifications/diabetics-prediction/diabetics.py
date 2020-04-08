# EDA packages
import eli5
import lime.lime_tabular
import lime
import shap
from sklearn.preprocessing import MinMaxScaler as Scaler
import missingno as no
import pandas as pd
import numpy as np

# Load ML packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#  Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# features needed
names = ["Num_of_Preg", "Glucose_Conc", "BP", "Skin_Thickness",
         "TwoHour_Insulin", "BMI", "DM_Pedigree", "Age", "Class"]
# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv', names=names)
df.head()
df.shape

# check for missing values
df.isna().sum()

# check the type of the datas
df.dtypes

# check for occurence of class
df.groupby('Class').size()

# plot the groups
df.groupby('Class').size().plot(kind="bar")

# check for correlation between input_data and labesl/outcomes
corr = df.corr()

# plot a heatmap
sns.heatmap(corr, annot=True)

# Data viz for entire datasets
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Descriptive analysis Transpose
df.describe().T

# Data cleaning
# Removing 0 sinces BP and can't be zero

# checking for minimum BP
df['BP'].min()

# checking for minimum BP
df['BMI'].min()

# Replace 0 with Median not Mean
df['BP'] = df['BP'].replace(to_replace=0, value=df['BP'].median())

# Recheck
df['BP'].min()

# Replace 0 with Median not Mean
df['BMI'] = df['BMI'].replace(to_replace=0, value=df['BMI'].median())

# checking for minimum TwoHour Insulin
df['TwoHour_Insulin'].min()

# Replace 0 with Median not Mean
df['TwoHour_Insulin'] = df['TwoHour_Insulin'].fillna(
    df['TwoHour_Insulin'].median())
# checking for minimum Glucose
df['Glucose_Conc'].min()

# Replace 0 with Median not Mean
df['Glucose_Conc'] = df['Glucose_Conc'].replace(
    to_replace=0, value=df['Glucose_Conc'].median())


df['Skin_Thickness'] = df['Skin_Thickness'].fillna(
    df['Skin_Thickness'].median())

# plot hist of all data again
df.hist(bins=50, figsize=(20, 15))
plt.show()

# checking for Null value or NO
no.bar(df)


# Feature Preparation

df.columns

# shaoe
df.shape

# slicing
df.iloc[:, 0:8]

# separate the Input data
Xfeatures = df.iloc[:, 0:8]

# sepate the labels
Ylabel = df['Class']


# scale the dataset
scaler = Scaler()
X = scaler.fit_transform(Xfeatures)


# check the shape of scaled data
X.shape

# features number
names[0:8]

# convert to dataframe from iloc
X = pd.DataFrame(X, columns=names[0:8])

# heads
X.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, Ylabel, test_size=0.2, random_state=42)

# check the splited shape
X_train.shape

# Building the Model with Logreg anf KNN
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# print the score
print("Accuracy Score of Logisitic::", logreg.score(X_test, y_test))

# make prediction
X_test.values[0]

# Prediction on A Single Sample
logreg.predict(np.array(X_test.values[0]).reshape(1, -1))

# Load the ML Interpreter explainer
# init Js for plot
shap.initjs()

explainer = shap.KernelExplainer(logreg.predict_proba, X_train)

shap_values = explainer.shap_values(X_test)

# plot
shap.summary_plot(shap_values, X_test)

# depency graph
shap.dependence_plot(0, shap_values[0], X_test)


# Glucose Concentration
shap.dependence_plot(1, shap_values[0], X_test)

# import lime
Ylabel.unique()

class_names = ['Non Diabetics', 'Diabetics']
df.columns

feature_names = ['Num_of_Preg', 'Glucose_Conc', 'BP', 'Skin_Thickness',
                 'TwoHour_Insulin', 'BMI', 'DM_Pedigree', 'Age']
# Create our Explainer ,a Tabular Explainer since it is a tabular data
explainer1 = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)


# Sample We Predicted
X_test.iloc[0]
# The Explainer Instance
exp1 = explainer1.explain_instance(
    X_test.iloc[0], logreg.predict_proba, num_features=8, top_labels=1)

# Show in notebook
exp1.show_in_notebook(show_table=True, show_all=False)

# sing ELI5

# Showing the Weight for our model
eli5.show_weights(logreg, top=10)

# Clearly Define Feature Names
eli5.show_weights(logreg, feature_names=feature_names,
                  target_names=class_names)


# Show Explaination For A Single Prediction
eli5.show_prediction(
    logreg, X_test.iloc[0], feature_names=feature_names, target_names=class_names)
