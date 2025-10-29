import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score
from tensorflow.keras.callbacks import EarlyStopping

def build_generator(latent_dim, output_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(36, activation='relu')(inputs)
    x = tf.keras.layers.Dense(36, activation='relu')(x)
    x = tf.keras.layers.Dense(36, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

def build_discriminator(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(36, activation='sigmoid')(inputs)
    x = tf.keras.layers.Dense(36, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(36, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

class GAN:
    def __init__(self, input_dim, latent_dim=100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.generator = build_generator(latent_dim, input_dim)
        self.discriminator = build_discriminator(input_dim)
        
        self.g_optimizer = tf.optimizers.Adam(0.0002, 0.5)
        self.d_optimizer = tf.optimizers.Adam(0.0002, 0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
    def train_step(self, real_data, batch_size):
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_data = self.generator(noise, training=True)
        
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            real_loss = self.loss_fn(real_labels, self.discriminator(real_data, training=True))
            fake_loss = self.loss_fn(fake_labels, self.discriminator(fake_data, training=True))
            d_loss = (real_loss + fake_loss) / 2
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            g_loss = self.loss_fn(real_labels, self.discriminator(fake_data, training=True))
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        return d_loss, g_loss

    def train(self, real_data, epochs, batch_size=64):
        for epoch in range(epochs):
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            
            d_loss, g_loss = self.train_step(real_batch, batch_size)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: [D loss: {d_loss.numpy():.4f}] [G loss: {g_loss.numpy():.4f}]")
        
        return self.generator

def augment_data(X_train, y_train, generator, latent_dim, num_samples):
    noise = tf.random.normal([num_samples, latent_dim])
    generated_samples = generator(noise).numpy()
    generated_labels = np.ones((num_samples,))
    X_augmented = np.vstack((X_train, generated_samples))
    y_augmented = np.hstack((y_train, generated_labels))
    return X_augmented, y_augmented

data_name = 'LC'
model_name = 'VAE'

def train_GAN():
    for desired_IR in [5,10,15,20,40,60]:
        data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
        X, y = data[:, :-1], data[:, -1]
        
        print('Imbalance rate:', np.sum(y) / (len(y) - np.sum(y)))
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        X_p = X[y == 1]
        gan = GAN(input_dim=X.shape[1])
        generator = gan.train(X_p, epochs=5000, batch_size=64)

        results = []
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        for train_idx, test_idx in sss.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            

            X_train_aug, y_train_aug = augment_data(X_train, y_train, generator, gan.latent_dim, num_samples=500)
            

            classifier = tf.keras.Sequential([
                # Hidden Layer 1: 64 (ReLU) + BatchNorm + Dropout (0.1)
                tf.keras.layers.Dense(64, input_dim=X_train.shape[1]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.1),
                
                # Hidden Layer 2: 32 (ReLU) + BatchNorm + Dropout (0.1)
                tf.keras.layers.Dense(32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.1),
                
                # Output Layer
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
  
            classifier.fit(
                X_train_aug, 
                y_train_aug, 
                epochs=200, 
                batch_size=32, 
                verbose=0,
                class_weight={0: 1, 1: desired_IR},
                validation_split=0.1,  
                callbacks=[early_stopping] 
            )

            y_pred_prob = classifier.predict(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            avg_precision = average_precision_score(y_test, y_pred_prob)
            brier_score = brier_score_loss(y_test, y_pred_prob)
            
            y_pred = np.argmax(classifier.predict(X_test), axis=1)
            sensitivity = recall_score(y_test, y_pred, pos_label=1)
            specificity = recall_score(y_test, y_pred, pos_label=0)
            g_means = np.sqrt(sensitivity * specificity)

            results.append([roc_auc, avg_precision, brier_score, g_means])
            results = np.array(results).T
            print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {avg_precision:.4f}, Brier Score: {brier_score:.4f}, G-Means: {g_means:.4f}")
        classifier.save(f'models/{data_name}{desired_IR}_{model_name}.h5')

if __name__ == '__main__':
    train_GAN()
