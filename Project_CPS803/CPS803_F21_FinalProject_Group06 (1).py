#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
## Naive Bayse
from sklearn.naive_bayes import GaussianNB
##Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#oversampling
from imblearn.over_sampling import SMOTE

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve


# # 5: Oversampling Technique to reduce the false negative rate for the models.

# **Synthetic Minority Oversample TEchnique (SMOTE)**

# OverSampling can be achieved with the SMOTE method where a new vector is generated between 2 existing datapoints. 
# 
# Applying this technique allows to massively increase the number of fraudulent transactions.


oversample = SMOTE()
X_resample, y_resample = oversample.fit_resample(X,y.values.ravel())

print('Number of total transactions--> before SMOTE upsampling: ', len(y), '-->after SMOTE upsampling: ', len(y_resample))
print('Number of fraudulent transactions--> before SMOTE upsampling: ', len(y[y.Class==1]), 
      '-->after SMOTE upsampling: ', np.sum(y_resample[y_resample==1]))

y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)

X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(24,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=10)

y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# **Note the Low value of False Negatives. The model is able to detect almost all fraudulent transactions on the full dataset.**
# 
# **Note the limited number of False Positives which means a lot less verification work (on legitimate transactions) for the fraud departement.**

acc = accuracy_score(y_test, y_pred.round())
prec = precision_score(y_test, y_pred.round())
rec = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())

### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset

# Confusion matrix on the whole dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# **Note the zero value of False Negatives. The model is able to detect all fraudulent transactions on the full dataset.**
# 
# **Note the limited number of False Positives which means a lot less verification work (on legitimate transactions) for the fraud departement.**

acc = accuracy_score(y, y_pred.round())
prec = precision_score(y, y_pred.round())
rec = recall_score(y, y_pred.round())
f1 = f1_score(y, y_pred.round())

model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_testset.append(model_results, ignore_index = True)
results_fullset

# **Note: All metrics are excellent for this last model.**