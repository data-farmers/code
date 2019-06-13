import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sammon import sammon

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

 
# READING DATA

df = pd.read_csv('../dataset/seeds.csv')

col1 = 'Area'
col2 = 'Perimeter'
col3 = 'Compactness'




# SAMMON MAPPING

# sammon(...) wants a Matrix
X = df.as_matrix(columns = [col1, col2, col3])

# By default, sammon returns a 2-dim array and the error E
[y, E] = sammon(X)




# PRINCIPAL COMPONENT ANALYSIS

# PCA wants a normalized Matrix
X_p = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_p)





# PLOT 

markers = ['o','^','p']
names = df['Name'].unique()

fig = plt.figure(figsize = (16, 5))



# SUBPLOT 1
ax0 = fig.add_subplot(131, projection='3d')

for i in range(3):
    ax0.scatter(df.loc[df['Name'] == names[i], col1],
                df.loc[df['Name'] == names[i], col2], 
                df.loc[df['Name'] == names[i], col3], 
                marker = markers[i], label = names[i])

ax0.set_xlabel(col1)
ax0.set_ylabel(col2)
ax0.set_zlabel(col3)
ax0.legend(title = "Raw data")



# SUBPLOT 2
ax1 = fig.add_subplot(132)

for i in range(3):
    indx = df.loc[df['Name'] == names[i]].index
    ax1.scatter(y[indx, 0], y[indx, 1], marker = markers[i], label = names[i])

ax1.legend(title = "Sammon mapping")



# SUBPLOT 3

ax2 = fig.add_subplot(133)

for i in range(3):
    indx = df.loc[df['Name'] == names[i]].index
    ax2.scatter(principalComponents[indx, 0], principalComponents[indx, 1], marker = markers[i], label = names[i])

ax2.legend(title = "PCA")



plt.savefig('../../data-farmers.github.io/img/sammon/sammonplot2.png')