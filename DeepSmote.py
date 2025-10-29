# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:57:16 2021

@author: coco
"""

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras
from keras import layers
import numpy as np
import os


encoder_dim = 40
data_name = 'LC'
model_name = 'DS'

def build_target(num_fea, LOSS='binary_crossentropy'):
    """
    Standard neural network training procedure with specified architecture.
    Hidden layers: [64 (ReLU), 32 (ReLU)].
    Includes Batch Normalization and Dropout (0.1) after each hidden layer.
    """
    model = Sequential()

    # --- Layer 1: Dense (64) + BatchNorm + Dropout (0.1) ---
    model.add(Dense(64, input_shape=(num_fea,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    # --- Layer 2: Dense (32) + BatchNorm + Dropout (0.1) ---
    model.add(Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    # --- Output Layer ---
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy'])
    return model


def cal_result(Cls, y_te, X_te):
    L = []
    y_prob = Cls.predict(X_te)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate evaluation metrics
    ROC = roc_auc_score(y_te, y_prob[:, 1])
    PR = average_precision_score(y_te, y_prob[:, 1])
    BS = brier_score_loss(y_te, y_prob[:, 1])
    
    # G-means calculation
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('sensitivity',sensitivity)
    print('specificity',specificity)
    
    G_means = np.sqrt(sensitivity * specificity)
    
    # Append results
    L.append(ROC)
    L.append(PR)
    L.append(BS)
    L.append(G_means)
    return np.array(L)


def DeepSmote():
    for desired_IR in [5,10,15,20,40,60]:
        Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
        X, y = Data[:, :-1], Data[:, -1]
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        DeepSmote_list = []

        ir = sum(y) / (len(y) - sum(y))
        print('imbalance rate:', ir)

        X_p = X[np.where(y == 1)]

        input_img = keras.Input(shape=(X.shape[1],))
        encoded = layers.Dense(encoder_dim, activation='sigmoid')(input_img)
        decoded = layers.Dense(X.shape[1], activation='sigmoid')(encoded)

        AE = keras.Model(input_img, decoded)
        AE.compile(optimizer='adam', loss='mse')

        noise = 0.001 * np.random.normal(size=np.shape(X_p))
        noisy_X = X_p + noise
        noisy_X = np.clip(noisy_X, 0.0, 1.0)

        AE.fit(noisy_X, X_p, epochs=100, batch_size=32, shuffle=True, verbose=1)

        encoder = keras.Model(input_img, encoded)

        encoded_input = keras.Input(shape=(encoder_dim,))
        decoder_layer = AE.layers[-1]
        decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

        for i in range(1):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
            for train_index, test_index in sss.split(X, y):
                X_tr = X[train_index]
                y_tr = y[train_index]
                X_te = X[test_index]
                y_te = y[test_index]

            rus_SMOTE = SMOTE(sampling_strategy=0.3)

            encoded_X = encoder.predict(X_tr)
            SMOTE_tr, SMOTE_y = rus_SMOTE.fit_resample(encoded_X, y_tr)
            decoded_X = decoder.predict(SMOTE_tr)

            Syn_Xp = decoded_X[np.where(SMOTE_y == 1)]
            Syn_yp = SMOTE_y[np.where(SMOTE_y == 1)]

            X_p = X_tr[np.where(y_tr == 1)]  # 原训练集正类样本
            X_re = AE.predict(X_p)  # 重构正类样本

            keep_list = []
            for sample_index in np.arange(len(Syn_Xp)):
                if Syn_Xp[sample_index] not in X_re:
                    keep_list.append(sample_index)
            keep_list = np.array(keep_list)

            Syn_Xp = Syn_Xp[keep_list]  # 仅保留生成的少数类样本
            Syn_yp = Syn_yp[keep_list]
            
            print('len(Syn_Xp)',len(Syn_Xp))
            print('len(X_tr)',len(X_tr))

            X_new = np.vstack((X_tr, Syn_Xp))
            y_new = np.hstack((y_tr, Syn_yp))
            y_new_ = to_categorical(y_new)

            clf = build_target(num_fea=X_tr.shape[1])
            clf.fit(X_new, y_new_, epochs=50, batch_size=20, shuffle=True,class_weight = {0:1, 1:desired_IR/2})

            DeepSmote_list.append(cal_result(clf, y_te, X_te))
        DeepSmote_list = np.array(DeepSmote_list)[0].T
        clf.save(f'models/{data_name}{desired_IR}_{model_name}.h5')

        # np.savetxt(f'results/PAKDD_DeepSmote_{desired_IR}.txt', DeepSmote_list)


DeepSmote()
