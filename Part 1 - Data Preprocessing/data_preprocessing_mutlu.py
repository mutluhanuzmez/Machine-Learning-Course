# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:54:10 2019

@author: Mutluhan
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis=0)
imputer = imputer.fit(features[:,1:3])
features[:,1:3] = imputer.transform(features[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_features = LabelEncoder()
features[:,0] = label_encoder_features.fit_transform(features[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()
label_encoder_labels = LabelEncoder()
labels = label_encoder_labels.fit_transform(labels)

#splitting data set into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_features = StandardScaler()
features_train = sc_features.fit_transform(features_train)
features_test = sc_features.transform(features_test)

