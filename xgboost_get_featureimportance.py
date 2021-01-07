# csv file : https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

import pandas as pd
import numpy
# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

numpy.seterr(divide='ignore', invalid='ignore')

# load data
df = pd.read_csv('./pima-indians-diabetes.csv')

# split data into X and y
X = df.iloc[:,0:8].values
y = df.iloc[:,8].values

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)

# fit model no training data
model = XGBClassifier()
model.fit(X, label_encoded_y)

# feature importance
print(model.feature_importances_)
