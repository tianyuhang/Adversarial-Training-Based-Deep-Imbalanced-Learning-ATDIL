import foolbox as fb
import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def PGD(X_P, y_P, targetModel, rho):
    '''
    rho: 攻击强度
    '''
    bounds = (-0.001, 1.001)
    fmodel = fb.TensorFlowModel(targetModel, bounds=bounds)
    attack = fb.attacks.L2ProjectedGradientDescentAttack(steps=5)  # 迭代次数

    raw, clipped, is_adv = attack(fmodel, X_P, y_P, epsilons=rho)  # 0.01-0.5
    return np.array(raw)

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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    G_means = np.sqrt(sensitivity * specificity)
    
    # Append results
    L.append(ROC)
    L.append(PR)
    L.append(BS)
    L.append(G_means)
    return np.array(L)

def build_target(num_fea, LOSS='binary_crossentropy'):
    """
    Standard neural network training procedure.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, input_shape=(num_fea,), activation='relu', kernel_initializer='RandomUniform', bias_initializer="RandomNormal"))
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy'])
    return model 

def Fsample(X, y, adData, adLable, sampling_strategy):
    '''
    使用正类对抗样本 adData 来扩充数据集
    '''
    rate = sampling_strategy / (1 + sampling_strategy)
    X = np.array(X)
    y = np.array(y)
    if set(y) != set([0, 1]):
        raise ValueError('样本类别标签不符合')    
  
    n_p = np.sum(y)
    n_n = len(y) - np.sum(y)   
    needed = int(((rate - 1) * n_p + rate * n_n) / (1 - rate))
    
    index = np.random.choice(len(adData), needed)
    newsamples = adData[index]
    newlables = adLable[index]
    X = np.vstack((X, newsamples))
    y = np.hstack((y, newlables))
    return X, y 


data_name = 'LC'
model_name = 'GAEOS'
def train_GAE():
    for desired_IR in [5,10,15,20,40,60]: 
        Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
        X, y = Data[:, :-1], Data[:, -1]

        print('imbalance rate:', sum(y) / (len(y) - sum(y)))
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        GAE_list = []
        
        for i in range(1):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=i)              
            for train_index, test_index in sss.split(X, y):
                X_tr = X[train_index]
                y_tr = y[train_index]
                X_te = X[test_index]
                y_te = y[test_index]
                
            y_tr_ = to_categorical(y_tr)
            target = build_target(num_fea=X_tr.shape[1])
            target.fit(X_tr, y_tr_, epochs=30, batch_size=10, shuffle=True)

            X_P = X_tr[np.where(y_tr == 1)]
            y_P = y_tr[np.where(y_tr == 1)]
            X_P = tf.convert_to_tensor(X_P, dtype=tf.float32)
            y_P = tf.convert_to_tensor(y_P, dtype=tf.int32)

            X_adv = PGD(X_P, y_P, target, 0.005)
            GAElabel = np.ones(len(X_adv))
            
            selected = np.random.randint(0, len(X_adv), 50)  # 增加的少数类样本
                    
            GAE_tr = np.vstack((X_tr, X_adv[selected]))
            GAE_y = np.hstack((y_tr, GAElabel[selected]))
            GAE_y_ = keras.utils.to_categorical(GAE_y)
            
            clf = build_target(num_fea=X_tr.shape[1])
            clf.fit(GAE_tr, GAE_y_, epochs=30, batch_size=10, shuffle=True,class_weight={0: 1, 1: desired_IR})
            
            GAE_list.append(cal_result(clf, y_te, X_te))
            print(cal_result(clf, y_te, X_te))
        GAE_list = np.array(GAE_list)[0].T
        clf.save(f'models/{data_name}{desired_IR}_{model_name}.h5')
        # np.savetxt(f'results/PAKDD_GAE_{desired_IR}.txt', GAE_list)

train_GAE()