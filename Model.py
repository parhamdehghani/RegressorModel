#!/usr/bin/env python
# coding: utf-8

# Used algorithm: XGBoost
# Parham Dehghani

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

# Here I import the dataset after perfoming some changes on the independent variables. I have removed the columns that do not have much contribution in the backpropagation; moved KWH to the last column as the dependent variable; moved METROMICRO, UR, and IECC_Climate_Pub columns to the first three columns as the categorical data for being encoded as the numbers using OneHotEncoder. The refined dataset named as "recs2009_public_removed.csv" then is imported as below. Then as dataset is a panda object, I seperate the numerical values to be input in the model building using 'values' attribute. The electrical consumption has also been moved to the last column and then is assigned to y as an array.


dataset = pd.read_csv('recs2009_public_removed.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encoding independent categorical data

# In this step, the first three columns will be converted to the numbers assuming that different categories within a column are independent from eachother. So the regression problem can be done properly. This stage of feature engineering is done using OneHotEncoder passing the columns that must be converted, the rest of the columns are untouched by adding remainder='passthrough'.


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Replacing null values

# There are some occurances of '.' in the data, so that it must be replaced by a numeric number. I use SimpleImputer class with 'most_frequent' strategy that replaces '.' with the most frequent value in each column.


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='.', strategy='most_frequent')
imputer.fit(X[:, 16:])
X[:, 16:] = imputer.transform(X[:, 16:])


# Splitting the dataset into the Training set and Test set
# This is the typical step in which the refined data must be split to two different sets. One is used for training the model, and the other which is not considered for the model building is inspected for the resulting accuracies of the predictions based on the trained model. I have chosen 20 % for the test set and 80 % for the training data.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature scaling
# The last step of the feature engineering is to use StandardScalar class to standardize all the features using default mean=0 and std=1 . The columns that are encoded for the categorical data are not considered for the scaling. The dependent variable is not also considered for the standardization. Note that scaling must be done based on training data, and the calculated parameters are used to scale down the test data. 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 16:] = sc.fit_transform(X_train[:, 16:])
X_test[:, 16:] = sc.transform(X_test[:, 16:])


# Training XGBoost on the Training set
# The chosen algorithm for this problem is XGBoost regressor that has been proved to be very efficient in many problems. I first instantiate this class and then apply the fit method on the regressor object. I also do not touch the hyperparameters of this algorithm as the default values are shown to be very effective as the result here is suggesting it.


from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)


# Calculating R^2 score
# The next step is to evaluate the trained model on the test set using R^2 score metric. As R^2 gets closer to 1, it shows more perfect predictions. Then, calculated R^2 score shows a perfect consistancy between the true values of the electrical consumptions and the predicted ones. In the last step, I will check for the stability of the accuracy results by splitting the training data to 10 different folds and check the accuracy for each of them.


from sklearn.metrics import r2_score
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('The resulting R^2 score is {} that shows a perfect set of predictions'.format(r2))


# Applying k-Fold Cross Validation
# Using cross_val_score function, the stability of the training accuracy is inspected by employing 10 different folds. The result shows that the mean accuracy of all the folds is appropriately in accordance with R^2 score with an acceptable std that shows almost 2 % differing accuracies for the folded training data. This result ensures the stability of the training task and suggests no overfitting considering the R^2 score value for the test set.


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Found accuracies for 10 different folds are: {}".format(accuracies))
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

