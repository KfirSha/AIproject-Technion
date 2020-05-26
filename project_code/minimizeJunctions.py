import numpy as np
import pandas as pd
import csv
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
from sklearn import neighbors
X = pd.read_csv("traffic_signals.csv")
X = X.as_matrix()
print(type(X))
Y = pd.read_csv("traffix_signals2.csv")
Y = Y.as_matrix()
New = []
nnn = []
ball_tree = neighbors.BallTree(X, leaf_size=2)
for i in range(len(Y)):
    if i not in New:
        nnn = Union(nnn,[i])
        ind = ball_tree.query_radius([Y[i]], r=50)
        a = ind[0].tolist()
        New = Union(New, a)
X_New = []
for i in range(len(nnn)):
    X_New.append(X[nnn[i]])
np.savetxt('2darray.csv', X_New, delimiter=',', fmt='%d')