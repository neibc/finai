# First XGBoost model for Pima Indians dataset
# from https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# csv file : https://github.com/bbangpan/finai/blob/main/winequality-red-binary.csv

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('winequality-red-binary.csv', skiprows=1, delimiter=",")
# split data into X and y
X = dataset[:,0:11]
Y = dataset[:,11]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
