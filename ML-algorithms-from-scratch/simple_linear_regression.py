import pandas as pd
from sklearn import model_selection
import math

# linear regression assumes a linear relationship between independent variables and dependent variable
# single input variable is known as simple linear regression
# y = b0 + b1 * x

# b1 = covariance / variance
# b0 = mean(y) - b1 * mean(x)

df = pd.read_csv("slr06.csv")
X = df["X"].values.tolist()
y = df["Y"].values.tolist()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)

def mean(vals):

    return sum(vals)/len(vals)

def variance(vals, mean):

    return sum([(x - mean)**2 for x in vals])

# covariance of two groups of numbers describes how those numbers change together
# correlation describes the relationship between two groups of numbers
# covariance describes the relationship between two or more groups of numbers
# covariance = sum(x(i)-mean(x)) * sum(y(i)-mean(y))

def covariance(x, mean_x, mean_y, y):

    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x) * (y[i]-mean_y)

    return covar

def coefficients(X, y):

    mean_x, mean_y = mean(X), mean(y)
    b1 = covariance(X, mean_x, mean_y, y)/variance(X, mean_x)
    b0 = mean_y - b1 * mean_x

    return [b0, b1]

def simple_linear_regression(X_train, X_test, y_train):

    b0, b1 = coefficients(X_train, y_train)
    preds = [b0 + b1 * x for x in X_test]
    
    return preds

def rmse(y_true, y_pred):

    error = 0.00

    for i in range(len(y_pred)):
        error += (y_true[i] - y_pred[i])**2 

    mean_error = error/len(y_true)

    return math.sqrt(mean_error)

preds = simple_linear_regression(X_train, X_test, y_train)
rmse = rmse(y_test, preds)

print(preds)
print("-"*50)
print(rmse)