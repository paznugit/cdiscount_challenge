# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 13:13:23 2015

@author: IGPL3460
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier)
from sklearn.tree import (DecisionTreeClassifier)
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM

# Load the training file
dataframe = pd.read_csv('../input/training_NG_MEQ_OK.csv',",")

#train, test = train_test_split(dataframe, test_size = 0.2)
train, test = train_test_split(dataframe, test_size = 0.2)

#On récupère les données de training et de test avec un y = category 1
X_train,y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test,y_test = test.iloc[:,:-1], test.iloc[:,-1]

# Parameters for the tree and the random forest
n_estimators = 100
max_depth = 40
min_samples_split = 1
class_weight = {0: 1,
                1: 6,
                2: 8,
                3: 10,
                4: 10,
                5: 20,
                6: 28,
                7: 33,
                8: 35,
                9: 37,
                10: 45}
params = {'n_estimators': n_estimators, 'max_depth': max_depth,
'min_samples_split': min_samples_split, 'class_weight': class_weight}
#class_weight={0:0.5, 1:0.5}

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Get rid of Nan values
X_train[np.isnan(X_train)] = 0.

#clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth=max_depth), n_estimators = n_estimators)
#clf = KNeighborsClassifier(3)
clf = RandomForestClassifier(**params)
#clf = RandomForestClassifier(n_estimators = n_estimators, max_depth= max_depth,
#min_samples_split= min_samples_split, class_weight= class_weight)
#clf = LogisticRegression(class_weight = class_weight)
#clf = SVC(gamma=2, C=1)
#clf = GaussianNB()
scores = cross_val_score(clf, X_train, y_train, cv=5)
#print 'BaggingClassifier:'
#print 'NaiveBayes:'
#print 'RandomForest:'
#print 'SVM'
print np.mean(scores), np.std(scores)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print clf.score(X_test,y_test)
confusion_mat = confusion_matrix(y_test,y_pred)
print "Confusion Matrix:"
print confusion_mat
#print clf.coef_
#print clf.intercept_