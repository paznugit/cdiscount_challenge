# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 14:50:18 2015

@author: IGPL3460
"""
import pandas as pd
import numpy as np

# Load the training file
dataframe = pd.read_csv('../input/training_NG_MEQ_OK.csv',",")

#iris_dataset = load_iris()


#X = pd.DataFrame(iris_dataset.data)
#y = (iris_dataset.target)

X = dataframe.as_matrix()[:,:-1]
y = dataframe.as_matrix()[:,-1]

classes = list(set(y))
palette = [(153. / 255, 79. / 255, 161. / 255),
           (255. / 255, 129. / 255, 1. / 255),
           (253. / 255, 252. / 255, 51. / 255)]  # HEXA

color_map = dict(zip(classes, palette))
colors = [color_map[y[i]] for i in xrange(len(y))]

# Affichage de la scatter matrix 
df = pd.DataFrame(X)

axeslist = pd.scatter_matrix(df, color=colors, diagonal='kde')
