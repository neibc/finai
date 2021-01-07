# csv file : https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 7

df_t = pd.read_csv('./pima-indians-diabetes.csv', header=None)
df = df_t.astype(np.float32)
del df_t

df_X = df.iloc[:,0:8].values
df_Y = df.iloc[:,8:9].values
del df

train_dataset, test_dataset, train_labels, test_labels = train_test_split(df_X, df_Y, shuffle = True, test_size = 0.3, random_state=seed)

nTrain = train_dataset.shape[0]
nTest = test_dataset.shape[0]

print ("Shape of (X_train, Y_train, X_test, Y_test)")
print (train_dataset.shape, train_labels.shape, test_dataset.shape, test_labels.shape)

# input data
df_t = pd.read_csv('./pima-indians-diabetes-valid.csv', header=None)
df = df_t.astype(np.float32)
del df_t

valid_dataset = df.iloc[:,0:8].values
valid_labels = df.iloc[:,8:9].values
del df

model = XGBClassifier(learning_rate=0.005)
print (model)

model.fit(train_dataset, train_labels.ravel())

# make prediction for test data
y_pred = model.predict(test_dataset)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(test_labels.ravel(), predictions)
print("Test Accuracy :%.5f%%" % (accuracy * 100.0))

# make prediction for validation data
y_pred = model.predict(valid_dataset)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(valid_labels.ravel(), predictions)
print("Valid Accuracy :%.5f%%" % (accuracy * 100.0))
