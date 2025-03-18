import foolbox as fb
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, brier_score_loss, average_precision_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf

t = [0.000001, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
SEI = []

for desired_IR in [5,10,15,20,40,60]:
    
    data_name = 'prosper'
    base = load_model(f'models/{data_name}{desired_IR}_base.h5')
    target_MINE = load_model(f'models/{data_name}{desired_IR}_MINE.h5')  
    Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
    X, y = Data[:,:-1], Data[:,-1]

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_te = X[test_index]
        y_te = y[test_index]
        X_P = X_te[np.where(y_te==1)]
        y_P = y_te[np.where(y_te==1)]
        X_P_= tf.convert_to_tensor(X_P, dtype=tf.float32)
        y_P_= tf.convert_to_tensor(y_P, dtype=tf.int32)

        X_te_= tf.convert_to_tensor(X_te, dtype=tf.float32)
        y_te_= tf.convert_to_tensor(y_te, dtype=tf.int32)


    bounds = (-0.0001, 1.0001)
    fmodel = fb.TensorFlowModel(base, bounds=bounds)
    attack = fb.attacks.L2ProjectedGradientDescentAttack(steps=50)



    recall_MINE_list = []

    # Iterate over different epsilon values (i)
    for i in t:
        # Get adversarial examples using the attack function
        raw, clipped, is_adv = attack(fmodel, X_P_, y_P_, epsilons=i)
        
        # Calculate recall for each model
        y_prob_MINE = target_MINE.predict(clipped)
        y_pred_MINE = np.argmax(y_prob_MINE, axis=1)
        recall_MINE = recall_score(y_P, y_pred_MINE)
        recall_MINE_list.append(recall_MINE)
        
    
    recall_MINE_list = np.array(recall_MINE_list)
    np.savetxt(f'security_results/{data_name}_{desired_IR}.txt', recall_MINE_list.T, delimiter='\t', comments='')

SMS = np.array(SEI)/0.15


