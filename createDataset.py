# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:57:16 2021

@author: coco
"""


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

Data = np.loadtxt('Data/credit2023.txt')
X, y = Data[:,:-1], Data[:,-1]


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

X, y = X_test, y_test



print('imbalance ratio:',(len(y)-sum(y))/sum(y))
IR = (len(y)-sum(y))/sum(y)

X_n = Data[np.where(y==0)]
X_p = Data[np.where(y==1)]

desired_IR_list = [5,10,15,20,40,60]

for desired_IR in desired_IR_list:
    print('Target imbalance ratio:', desired_IR)

    if desired_IR < IR:

        desired_n = int(len(X_p) * desired_IR)
        ix_n = np.random.choice(len(X_n), desired_n, replace=False)
        X_n_new = X_n[ix_n]
        X_p_new = X_p  

    elif desired_IR > IR:
     
        desired_p = int(len(X_n) / desired_IR)
        ix_p = np.random.choice(len(X_p), desired_p, replace=False)
        X_p_new = X_p[ix_p]
        X_n_new = X_n  

    else:

        X_n_new = X_n
        X_p_new = X_p


    D = np.vstack((X_n_new, X_p_new))
    np.random.shuffle(D) 


    np.savetxt(f'ImbalanceData/credit2023_{desired_IR}.txt', D)

