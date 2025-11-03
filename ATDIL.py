# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, brier_score_loss, average_precision_score, roc_auc_score
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling1D, Reshape, Multiply, Input
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras import layers
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf


def cal_result(cls, y_test, X_test):
    """
    Calculate and return evaluation metrics for the model on the test set:
    - ROC AUC
    - PR AUC
    - Brier Score (BS)
    - G-means
    """
    results = []
    y_prob = cls.predict(X_test)  # Model predicted probabilities
    y_pred = np.argmax(y_prob, axis=1)  # Convert to class predictions

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])

    # Calculate PR AUC
    pr_auc = average_precision_score(y_test, y_prob[:, 1])

    # Calculate Brier Score (BS)
    # Note: Brier Score uses predicted probabilities and true labels
    brier_score = brier_score_loss(y_test, y_prob[:, 1])

    # Calculate G-means
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    g_means = np.sqrt(tpr * (1 - fpr)).max()

    # Append results to the list
    results.extend([roc_auc, pr_auc, brier_score, g_means])
    return np.array(results)


def scheduler(epoch, lr):
    """
    Learning rate scheduler function.
    Reduces learning rate after 50 epochs.
    """
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


class AdversarialTraining:
    """
    Adversarial Training class for training a model with adversarial examples.
    """
    def __init__(self):
        self.attacking_strength = 0.001
        self.num_features = X.shape[1]
        self.encoding_dim = 30
        self.target_model = self.build_target_model()
        self.num_iterations = 100
        self.batch_size = 32
        self.autoencoder = self.build_autoencoder()

    def build_target(self):
        model = Sequential()
        model.add(Input(shape=(Data.shape[1] - 1,)))  # 输入维度不变
    
        # Hidden Layer 1: 64 -> BN -> ReLU -> Dropout 0.1
        model.add(Dense(64, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
    
        # Hidden Layer 2: 32 -> BN -> ReLU -> Dropout 0.1
        model.add(Dense(128, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
    
        # Output: binary classification
        model.add(Dense(2, activation='sigmoid'))
    
        # Optimizer & compile 
        model.compile(
            # loss='binary_crossentropy',
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=1e-3),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def build_autoencoder(self):
        """
        AAE
        """
        def squeeze_excite_block(input_tensor, ratio=16):
            channel = input_tensor.shape[-1]
            se = layers.GlobalAveragePooling1D()(layers.Reshape((1, channel))(input_tensor))  # Squeeze
            se = layers.Dense(channel // ratio, activation='relu')(se)  # Excitation
            se = layers.Dense(channel, activation='sigmoid')(se)  # Excitation
            se = layers.Reshape((channel,))(se)  # reshape
            output_tensor = layers.Multiply()([input_tensor, se])  
            return output_tensor
    
        # input
        input_img = keras.Input(shape=(self.num_features,))
    
        # encoder
        encoded = layers.Dense(64, activation='relu')(input_img)
        encoded = squeeze_excite_block(encoded)  # add SE layer
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
    
        # decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = squeeze_excite_block(decoded)  # add SE layer
        decoded = layers.Dense(self.num_features, activation='sigmoid')(decoded)
    
        AE = keras.Model(input_img, decoded)
    
        def custom_loss(y_true, y_pred):
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(y_pred - y_true), axis=1)
            pos = self.target_model(y_pred)[:, 1]
            adversarial_loss = self.attacking_strength * pos
            return reconstruction_loss + adversarial_loss
    
        AE.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss)
        return AE
        

    def train(self, X_train, y_train, base_model=None):
        """
        Train the target model with adversarial examples.
        """
        y_train_categorical = to_categorical(y_train)
        
        X_positive = X_train[np.where(y_train == 1)]
       
        # Add noise to positive samples
        noise = 0.01 * np.random.normal(size=np.shape(X_positive))
        noisy_positive = X_positive + noise
        noisy_positive = np.clip(noisy_positive, 0.0, 1.0) 

        # Train the target model
        self.target_model.fit(X_train, y_train_categorical, epochs=10, batch_size=64, shuffle=True, class_weight={0: 1, 1: 1})
        
        # Evaluate the original model
        self.original_metrics = cal_result(self.target_model, y_test, X_test)

        # Train the autoencoder
        self.autoencoder.fit(noisy_positive, X_positive, epochs=100, batch_size=10, shuffle=True, verbose=0)
        
        
        # Adversarial training loop
        for epoch in range(self.num_iterations):
            # Generate adversarial examples
            adversarial_examples = self.autoencoder.predict(X_positive + np.random.normal(0, 0.001, X_positive.shape))
            adversarial_examples = np.clip(adversarial_examples, 0.0, 1.0)
            y_adversarial = np.ones(len(adversarial_examples))
            y_adversarial = to_categorical(y_adversarial)
            
            # Randomly select a batch of adversarial examples
            idx = np.random.randint(0, len(adversarial_examples), 10)
            X_adv_batch = adversarial_examples[idx]
            y_adv_batch = y_adversarial[idx]       

            # Create a subset of the training data
            subset_ratio = 0.2  # 30% of the data as a subset
            X_batch, _, y_batch, _ = train_test_split(X_train, y_train_categorical, test_size=(1 - subset_ratio))

            # Combine the subset with adversarial examples
            X_batch = np.vstack((X_batch, X_adv_batch))
            y_batch = np.vstack((y_batch, y_adv_batch))

            # Train the target model with the combined data
            lr_scheduler = LearningRateScheduler(scheduler)
            self.target_model.fit(X_batch, y_batch, shuffle=True, verbose=0, class_weight={0: 1, 1: 1})

            # Periodically retrain the autoencoder
            if epoch % 20 == 0:
                self.autoencoder.fit(noisy_positive, X_positive, shuffle=True, verbose=0)
                    
            # Evaluate the model periodically
            if epoch % 1 == 0:  
                print(f'Epoch: {epoch}')
                current_metrics = cal_result(self.target_model, y_test, X_test)                        
                print(current_metrics)
            self.best = current_metrics 

        return self.target_model
# Main execution
if __name__ == '__main__':

    data_name = 'credit2023'
    for desired_IR in [5,10,15,20,40,60]:
        # Load dataset
        Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
        X, y = Data[:, :-1], Data[:, -1]
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
         
        for seed in range(10):
            # Split the data into training and testing sets
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)              
            for train_index, test_index in sss.split(X, y):
                X_train = X[train_index]
                y_train = y[train_index]    
                X_test = X[test_index]
                y_test= y[test_index]
            
            adversarial_trainer = AdversarialTraining()
            trained_model = adversarial_trainer.train(X_train, y_train, base_model=None)
            trained_model.save(f'models/{data_name}{desired_IR}_MINE.h5')
            print(adversarial_trainer.best)
