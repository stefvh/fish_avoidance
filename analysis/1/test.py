import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# read from csv file fish_behaviors_e=0.csv
fish_behaviors = pd.read_csv('fish_behaviors_e=0.csv')
fish_behaviors.loc[fish_behaviors['behavior'] == 'EVADE', 'target'] = 0
fish_behaviors.loc[fish_behaviors['behavior'] == 'EXPLORE', 'target'] = 1
fish_behaviors.loc[fish_behaviors['behavior'] == 'FREEZE', 'target'] = 2



iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

print(y)
print(names)
print(feature_names)

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Modifying the dataset
X[y==1,2] = X[y==1,2] + 0.3
# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=2)
# Decreasing the train set to make things more interesting
X_train = X_train[30:,:]
Y_train = Y_train[30:,:]

