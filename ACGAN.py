import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os

# 数据集列表
datasets = ["prosper_40"]

# 创建结果保存目录
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, input_dim):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        input_combined = torch.cat((noise, label_embedding), dim=1)
        return self.model(input_combined)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.validity_layer = nn.Linear(256, 1)
        self.label_layer = nn.Linear(256, num_classes)
    
    def forward(self, x):
        features = self.model(x)
        validity = torch.sigmoid(self.validity_layer(features))
        label = torch.softmax(self.label_layer(features), dim=1)
        return validity, label

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(30, 30),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(30, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 生成少数类样本
def generate_synthetic_samples(generator, num_samples, latent_dim, target_label):
    noise = torch.randn(num_samples, latent_dim)
    labels = torch.full((num_samples,), target_label, dtype=torch.long)
    return generator(noise, labels).detach().numpy()

# 主循环
for dataset_name in datasets:
    # 加载数据
    data = np.loadtxt(f"ImbalanceData/{dataset_name}.txt")
    X, y = data[:, :-1], data[:, -1]
    
    # 数据归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # 数据分割
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 初始化模型参数
    latent_dim = 100
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    # 初始化模型
    generator = Generator(latent_dim, num_classes, input_dim)
    discriminator = Discriminator(input_dim, num_classes)
    classifier = Classifier(input_dim, num_classes)
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(classifier.parameters(), lr=0.001)
    
    # 损失函数
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    
    # ACGAN 训练
    epochs = 200
    batch_size = 64
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train_tensor[idx]
        real_labels = y_train_tensor[idx]
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        
        # 训练生成器
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, latent_dim)
        sampled_labels = torch.randint(0, num_classes, (batch_size,))
        gen_data = generator(noise, sampled_labels)
        validity, pred_label = discriminator(gen_data)
        g_loss = adversarial_loss(validity, valid) + auxiliary_loss(pred_label, sampled_labels)
        g_loss.backward()
        optimizer_G.step()
        
        # 训练判别器
        optimizer_D.zero_grad()
        validity_real, pred_label_real = discriminator(real_data)
        validity_fake, pred_label_fake = discriminator(gen_data.detach())
        d_loss_real = adversarial_loss(validity_real, valid) + auxiliary_loss(pred_label_real, real_labels)
        d_loss_fake = adversarial_loss(validity_fake, fake) + auxiliary_loss(pred_label_fake, sampled_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
    
    # 增强数据集
    minority_class = 1
    synthetic_samples = generate_synthetic_samples(generator, 50, latent_dim, minority_class)
    X_train_augmented = np.vstack((X_train, synthetic_samples))
    y_train_augmented = np.hstack((y_train, np.full(50, minority_class)))
    
    train_dataset = TensorDataset(torch.tensor(X_train_augmented, dtype=torch.float32),
                                  torch.tensor(y_train_augmented, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    
    class_weights = torch.tensor([1.0, 30.0], dtype=torch.float32)  # 假设类别 0 权重为 1，类别 1 权重为 10
    
    # 训练分类器
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer_C.zero_grad()
            outputs = classifier(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer_C.step()
    
    # 测试分类器
    classifier.eval()
    with torch.no_grad():
        y_pred_proba = torch.softmax(classifier(X_test_tensor), dim=1).numpy()
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 计算评估指标
    auc_score = roc_auc_score(y_test, y_pred_proba[:, minority_class])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, minority_class])
    pr_auc = auc(recall, precision)
    brier_score = brier_score_loss(y_test, y_pred_proba[:, minority_class])
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    gmeans = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    
    # 保存结果
    results = [auc_score, pr_auc, brier_score, gmeans]
    results_path = os.path.join(results_dir, f"{dataset_name}_ACGAN_results.txt")
    np.savetxt(results_path, np.array(results).reshape(-1, 1), fmt="%.4f")
    print(f"Results for {dataset_name} saved to {results_path}.")