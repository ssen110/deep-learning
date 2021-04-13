# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:39:39 2021

@author: asus-pc
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


df = pd.read_csv('Churn_Modelling.csv')

X = df.iloc[: , 3:13].values
Y = df.iloc[:, 13].values


# encode the categorical values into number
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le_1 = LabelEncoder()
le_2 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[: , 1])
X[:, 2] = le_1.fit_transform(X[: , 2])
# to make more than 2 values from a category to binary as otherwise it can fall into traps of weight assign

one_1 = ColumnTransformer([("ANYNAME", OneHotEncoder(), [1])], remainder = 'passthrough')
X = one_1.fit_transform(X) 
X = X[:, 1:] # to resolve the dummy variable trap

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = True)


## Scalling the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Making the ANN
# importing packages
import keras

from keras.models import Sequential
from keras.layers import Dense

# initialize layers

clf = Sequential()  # clf is the future classifier

#adding input layer and hidden in the ANN

clf.add(Dense(6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))

## Hidden layer

clf.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Final Layer

clf.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#### Compelling the ANN

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # optimizer like Gradient D  , loss function

# Fitting the ANN 

clf.fit(X_train, Y_train, batch_size = 10, epochs =100)

#prediction 

Y_pred = clf.predict(X_test)

Y_pred = (Y_pred > 0.5) # Y_pred > 0.5 = 1 else 0

# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)




















