import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, brier_score_loss
from sklearn.metrics import auc as calculate_auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
import os


results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        return self.disc(data)



def build_target(input_dim, LOSS='binary_crossentropy'):
    model = Sequential()
    
    # Hidden Layer 1: 64 + BatchNorm + ReLU + Dropout (0.1)
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    # Hidden Layer 2: 32 + BatchNorm + ReLU + Dropout (0.1)
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    # Output Layer
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy'])
    return model

data_name = 'LC'
model_name = 'SGAN'

for desired_IR in [5,10,15,20,40,60]:

    Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
    X, y = Data[:, :-1], Data[:, -1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


    sm = SMOTE(random_state=0)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


    minority_class_label = 1
    minority_indices = np.where(y_train_res == minority_class_label)[0]
    X_train_minority = X_train_res[minority_indices]
    y_train_minority = y_train_res[minority_indices]

    tensor_x = torch.Tensor(X_train_minority)
    tensor_y = torch.Tensor(y_train_minority)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    input_dim = X_train.shape[1]
    hidden_dim = 64
    z_dim = input_dim
    n_epochs = 100
    lr = 0.0001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = Generator(z_dim, hidden_dim).to(device)
    disc = Discriminator(input_dim, hidden_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()


    for epoch in range(n_epochs):
        for real, _ in dataloader:
            real = real.to(device)
            cur_batch_size = real.size(0)


            disc_opt.zero_grad()
            noise = torch.randn(cur_batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_fake_pred = disc(fake.detach())
            disc_real_pred = disc(real)
            disc_loss = (criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)) +
                         criterion(disc_real_pred, torch.ones_like(disc_real_pred))) / 2
            disc_loss.backward()
            disc_opt.step()


            gen_opt.zero_grad()
            fake = gen(noise)
            disc_fake_pred = disc(fake)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

    noise = torch.randn(X_train_minority.shape[0], z_dim).to(device)
    fake_samples = gen(noise).detach().cpu().numpy()

    X_train_final = np.vstack([X_train_res, fake_samples])
    y_train_final = np.hstack([y_train_res, np.ones(fake_samples.shape[0])])


    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_final)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_encoded, num_classes=2)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=2)


    target_model = build_target(X_train.shape[1])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    

    target_model.fit(
        X_train_final, 
        y_train_categorical, 
        epochs=200, 
        batch_size=32, 
        verbose=1, 
        validation_split=0.1, #
        callbacks=[early_stopping]
    )


    y_pred_prob = target_model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)


    auc_score = roc_auc_score(y_test_encoded, y_pred_prob[:, 1])
    precision, recall, _ = precision_recall_curve(y_test_encoded, y_pred_prob[:, 1])
    pr_auc = calculate_auc(recall, precision)
    brier_score = brier_score_loss(y_test_encoded, y_pred_prob[:, 1])
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_prob[:, 1])
    # G-means using the best threshold from ROC curve
    gmeans = np.sqrt(tpr * (1 - fpr)).max()


    target_model.save(f'models/{data_name}{desired_IR}_{model_name}.h5')
    # result_path = os.path.join(results_dir, f"PAKDD_{desired_IR}_SGAN_results.txt")
    # with open(result_path, "w") as f:
    #     f.write(f"{auc_score:.4f}\n")
    #     f.write(f"{pr_auc:.4f}\n")
    #     f.write(f"{brier_score:.4f}\n")
    #     f.write(f"{gmeans:.4f}\n")
