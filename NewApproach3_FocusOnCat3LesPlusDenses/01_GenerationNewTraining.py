# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 11:34:14 2015

@author: IGPL3460
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Categories 3 à analyser
categories3ToAnalyse = {'1000010653': 13.9,
'1000015309': 6.5,
'1000004085': 5.4,
'1000004079': 3.6,
'1000010667': 2.7,
'1000012993': 1.9,
'1000010170': 1.6,
'1000010533': 1.5,
'194': 1.4,
'1000008094': 1.2,
}

percentOther = 100 - np.sum(categories3ToAnalyse.values())

numberOfLine = 4000

#Modification du dictionnaire categories
classe = 1
for key in categories3ToAnalyse.keys():
    percent = categories3ToAnalyse[key]
    number = numberOfLine * percent/100
    categories3ToAnalyse[key] = [0,number,classe]
    classe += 1
other = [0,percentOther * numberOfLine/100]

categoryPattern = "^[0-9]+;(?P<cat1>[0-9]+);(?P<cat2>[0-9]+);(?P<cat3>[0-9]+);"

csv_input = '../input/training.csv'
csv_cible = '../input/training_NG.csv'

cibleBucket = 11
compteurBucket = 0

with open(csv_cible, 'w') as outfile:
    with open(csv_input, 'r') as infile:
        i = 0
        for line in infile:
            if i ==0:
                outfile.write(line)
                i += 1
                continue;
            
            #if i > 1000:
             #   break

            # Retrieve category 1
            match=re.search(categoryPattern, line)
            cat1 = match.group('cat1')
            cat2 = match.group('cat2')
            cat3 = match.group('cat3')
            
            cat = cat3
            
            if cat in categories3ToAnalyse:
                
                # Gestion aléatoire du sample
                rand = np.random.random_integers(1,5)
                if rand < 5:
                    continue;            
                
                catNumbers = categories3ToAnalyse[cat]
                if catNumbers[0] > catNumbers[1]: 
                    continue
                
                catNumbers[0] += 1
                newline = string.replace(line,cat,str(catNumbers[2]))
                outfile.write(newline)
                i += 1 
                if catNumbers[0] > catNumbers[1]: 
                    compteurBucket += 1
            else:
                if other[0] > other[1]: 
                    continue

                rand = np.random.random_integers(1,100)
                if rand < 100:
                    continue

                other[0] += 1
                newline = string.replace(line,cat,'0')
                outfile.write(newline)
                i += 1 
                if other[0] > other[1]: 
                    compteurBucket += 1

            if compteurBucket == cibleBucket:
                break
        print "nb ligne = " + str(i)
           
# Load the training file
training_dataframe = pd.read_csv('../input/training_NG.csv',";")

nbLine = len(training_dataframe.index)

#############################
# Récupération de l'etiquette
#############################
y = training_dataframe.Categorie3.values

#############################
# Generation de l'input TF_IDF lié au vocabulaire de category
#############################

# Load the rayon file
'''rayon_dataframe = pd.read_csv('../input/rayon.csv',";")
# Creation of the vocabulary
listeCategoryCode = rayon_dataframe['Categorie3']
listeCategoryName = rayon_dataframe['Categorie3_Name']

vocab = {}
j = 0
for (cat,name) in zip(listeCategoryCode,listeCategoryName):
    if True:#str(cat) in categories3ToAnalyse:
        words = name.split()
        # for each word in the line:
        for word in words:
            word = word.lower()
            if word not in vocab:
                vocab[word] = j
                j += 1'''
                
# Fit TFIDF
descriptions = training_dataframe['Description']

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english', decode_error= 'ignore', lowercase = True)
        
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
#X = X.todense()

np.savetxt('../input/training_NG_MEQ_OK.csv',data,delimiter=',',fmt='%1.1f')                
