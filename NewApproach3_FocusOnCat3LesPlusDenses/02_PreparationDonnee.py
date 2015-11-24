# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:47:12 2015

@author: IGPL3460
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the training file
training_dataframe = pd.read_csv('../input/training_NG.csv',";")

nbLine = len(training_dataframe.index)

#############################
# Récupération de l'etiquette
#############################
y = training_dataframe.Categorie3.values

#############################
# Generation de l'input marque
#############################
'''listeMarque = training_dataframe['Marque']
dicoMarque = {}
i = 0
for marque in listeMarque:
    if marque not in dicoMarque:
        dicoMarque[marque] = i
        i += 1

X_marque = np.zeros((nbLine,i))

line = 0
for marque in listeMarque:
    indice = dicoMarque[marque]
    X_marque[line,indice] = 1
    line += 1'''

#############################
# Generation de l'input TF_IDF lié au vocabulaire de category
#############################

vocab = {'portable' : 0,
         'coque' : 1,
         'batterie' : 2,
         'téléphone' : 3,
         'étui' : 4}
         
vocab = {'coque' : 0,
         'bumper' : 1,
         'facade' : 2,
         'téléphone' : 3,
         'valise' : 4,
         'babage' : 5,
         'bijou' : 6,
         'ordinateur' : 7,
         'tablette' : 8,
         'pages' : 9,
         'édition' : 10}
         
# Fit TFIDF
descriptions = training_dataframe['Description']

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english', vocabulary = vocab, decode_error= 'ignore', lowercase = True)
        
tfv.fit(descriptions)

X =  tfv.transform(descriptions)   

#############################
# Nettoyage de la dataframe
#############################

# Champ inutile pour l'apprentissage dans un premier temps
training_dataframe = training_dataframe.drop('Identifiant_Produit', axis=1)
training_dataframe = training_dataframe.drop('Categorie1', axis=1)
training_dataframe = training_dataframe.drop('Categorie2', axis=1)
training_dataframe = training_dataframe.drop('Categorie3', axis=1)
training_dataframe = training_dataframe.drop('Description', axis=1)
training_dataframe = training_dataframe.drop('Libelle', axis=1)
training_dataframe = training_dataframe.drop('Marque', axis=1)
training_dataframe = training_dataframe.drop('Produit_Cdiscount', axis=1) 

#############################
# Generation de la matrice finale
#############################

data = np.hstack((X.todense(),np.reshape(y,(nbLine,1))))

np.savetxt('../input/training_NG_MEQ_OK.csv',data,delimiter=',',fmt='%10.1f')


  

 

            
   