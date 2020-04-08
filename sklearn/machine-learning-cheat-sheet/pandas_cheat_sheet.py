import pandas as pd

'''

# read the data
data = pd.read_csv('')
data = pd.read_csv(
    "https://github.com/Shreyas3108/Weather/raw/master/weather.csv")

'''
data.head()

# get the data Info
data.info()


# perfom some EDA
data.isnull().values.any()
# count all the labels
count_classes = pd.value_counts(data['Class'], sort=True)

# data shape
data.shape

# describe the data
data.describe()

'''
# features needed
names = ["Num_of_Preg", "Glucose_Conc", "BP", "Skin_Thickness",
         "TwoHour_Insulin", "BMI", "DM_Pedigree", "Age", "Class"]
# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv', names=names)

'''
# Feature Preparation
df.columns

# slicing
df.iloc[:, 0:8]

'''
# check for missing values
df.isna().sum()
# remove missing values
data = data.dropna()

# check the null data and sum it
data.isnull().sum()

# drop null data
data = data.dropna()
'''

'''
# counts
# count the value of RainToday Feature
data.RainToday.value_counts()

'''

# convert to dataframe
X = pd.DataFrame(X, columns=names[0:8])

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

# checking for minimum BP
df['BP'].min()
# Replace 0 with Median not Mean
df['BP'] = df['BP'].replace(to_replace=0, value=df['BP'].median())

# checking for Null value or NO
no.bar(df)

'''

sns.set_style('whitegrid')
sns.countplot(x='target', data=data, palette='RdBu_r')

# plot of a graph
graph = sns.countplot(x='RainToday', data=data)

# distribution plot of minTemp
graph = sns.distplot(data['MinTemp'], bins=40)

# distribution plot of rainfall
graph = sns.distplot(data['Rainfall'])
'''

'''

# visualization of bedroom
data.bedrooms.value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

'''

'''
# data conversions
data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == "Yes" else 0)

train = data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
              'WindDir3pm', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm', 'RainToday', 'RISK_MM']]

data['RainTomorrow'] = data['RainTomorrow'].apply(
    lambda x: 1 if x == "Yes" else 0)

label = data['RainTomorrow']

'''
