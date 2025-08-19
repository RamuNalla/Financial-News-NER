# Implement a logistic regression model that given a set of predictors and a response variable (0/1), predict the probability of success

# You don't need to write functional code. The idea is to think about how you would implement a class for this task - some talking points: 

# 1. What would be the variables you will store and for what purpose?

# 2. What functions will you need to implement and what arguments will they take as inputs?

 

import numpy as np
import pandas as pd

np.random.seed(42)


#class LogisticRegression(self, X, y, lr, n_ierations):

def sigmoid(x):
    return (1/(1+np.exp(x)))

def cost_function(y, y_pred):
    return (y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def logisticregression(X, y, lr=0.01, n_iters):

    n_samples, n_features = X.shape[0], X.shape[1]
    cost_history = []
    X_bias = np.c_[np.ones(n_samples), X]               # 50 x 6

    weights = np.random.rand(n_features)                    # [6, ]

    for i in range(n_iters):
        y_prob = sigmoid(X_bias * weights)              # [50, ]

        y_prob = y_prob.flatter()
        y = y.flatten()
        error = y - y_prob                              #[50,1]

        gradients = (1/X.shape[0])*(X_bias.T * error)       # error * X_bias  [6,1]

        weights -= lr * gradients

        cost_history.append()

    


    
    







num_rows = 50

num_features = 5

X = np.random.randn(num_rows, num_features)

Y = (np.random.rand(num_rows) > 0.5).astype(int)

# Display the generated data

print(X)

print(Y)