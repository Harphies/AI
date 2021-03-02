## Project zoo on data cleaning

# Data Cleaning involve two categories

- Dealing with Missing data
- Dealing with Outliers

# Techniques for deaing with Missing data

- Deletion or listwise deletion
- Imputation
  - Replacing with an average (mean)
  - Replacing with mean, median, mode
  - Interpolation from nearby values
  - Build a model to predict a missing value
- Hot-deck imputation
  - Last observation carried forward
- Mean substitution
  - It weakens correlation
- Regression
  - Build a model to predict other columns based on some columns
  - It strenthens correlations

`Ouliers: A data point that differs significantly from other data points in the same datasets.`

# Techniques for dealing with outliers:

- Identifying the ouliers
- Coping with outliers

# identifying Outiers

- Identify the distance from the mean
- identify the distance from the fitted line (regression line)

# coping with Outliers:

- Drop
- cap/Floor
- Set to mean (replace with mean of the data in that column)

# Feature Engineering and working with data

- Features selection
- Features extraction
- Features combination
- Features learning
- Dimentionality reduction Algorithm
  - Principal Component Analysis (PCA)
  - Autoencoder
