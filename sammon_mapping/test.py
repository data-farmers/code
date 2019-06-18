import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
#X = df.iloc[:,:7].values


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

fig = plt.figure(figsize = (18, 6))



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
ax0.set_title("Raw data")
ax0.legend()



# SUBPLOT 2
ax1 = fig.add_subplot(132)

for i in range(3):
    indx = df.loc[df['Name'] == names[i]].index
    ax1.scatter(y[indx, 0], y[indx, 1], marker = markers[i], label = names[i])


ax1.set_title("Sammon mapping")
ax1.legend()



# SUBPLOT 3

ax2 = fig.add_subplot(133)

for i in range(3):
    indx = df.loc[df['Name'] == names[i]].index
    ax2.scatter(principalComponents[indx, 0], principalComponents[indx, 1], marker = markers[i], label = names[i])

ax2.set_title("PCA")
ax2.legend()



plt.savefig('../../data-farmers.github.io/img/sammon/sammonplot2.png')
