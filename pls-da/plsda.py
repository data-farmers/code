#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:36:34 2019

@author: alfonso
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler






 
# READING DATA

df = pd.read_csv('../dataset/seeds.csv')


X = df.iloc[:,0:7].values
y_categorical = df['Name'].unique()
classes = df['Type']


# Plot settings
cols = ['blue','orange','green']
mks = ['o','^','p']
colorlist = [cols[i-1] for i in classes]
markerlist = [mks[i-1] for i in classes]

# Making the dummy Y response matrix
y = np.zeros(shape=(df.shape[0], 3))
for i in range(df.shape[0]):
    y[i, classes[i] - 1] = 1


plsr = PLSRegression(n_components=2, scale=False) # <1>
plsr.fit(X, y)
scores = plsr.x_scores_



# PCA wants a normalized Matrix
X_p = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_p)




fig = plt.figure(figsize = (11, 5))

ax0 = fig.add_subplot(121)
for i in range(len(y_categorical)):
    indx = df.loc[df['Name'] == y_categorical[i]].index
    ax0.scatter(scores[indx,0], scores[indx,1], marker = mks[i], label = y_categorical[i])

ax0.set_xlabel('Scores on LV 1')
ax0.set_ylabel('Scores on LV 2')
ax0.set_title('PLS-DA')
ax0.legend(loc = 'upper right')

ax1 = fig.add_subplot(122)
for i in range(len(y_categorical)):
    indx = df.loc[df['Name'] == y_categorical[i]].index
    ax1.scatter(principalComponents[indx, 0], principalComponents[indx, 1], marker = mks[i], label = y_categorical[i])

ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')
ax1.set_title('PCA')
ax1.legend(loc = 'upper right')

plt.savefig('../../data-farmers.github.io/img/pls-da/plot1.png')