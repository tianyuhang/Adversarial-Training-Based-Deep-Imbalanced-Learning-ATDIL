import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix
import os
from sklearn.utils.class_weight import compute_class_weight


# 数据集列表
datasets = ["home_20", "home_40", "home_60"]

# 确保结果保存目录存在
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, noise_dim, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, noise, feature_weights):
        gen_data = self.model(noise)
        return gen_data * feature_weights

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.model(data)

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        return self.model(data)

data_name = 'credit'
model_name = 'DFG'
# 遍历每个数据集
for desired_IR in [5,10,15,20,40,60]: 
    # 加载数据
    Data = np.loadtxt(f'ImbalanceData/{data_name}_{desired_IR}.txt')
    X, y = Data[:, :-1], Data[:, -1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 计算类别权重
    class_weights = torch.tensor([1.0, desired_IR], dtype=torch.float32)  # 假设类别 0 权重为 1，类别 1 权重为 10
    
    # 初始化模型
    input_dim = X_train.shape[1]
    noise_dim = 10
    num_classes = len(np.unique(y_train))
    generator = Generator(noise_dim, input_dim)
    discriminator = Discriminator(input_dim)
    classifier = Classifier(input_dim, num_classes)

    # 优化器
    lr = 0.001
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    c_optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # 特征权重优化器
    feature_weights = torch.ones(input_dim, requires_grad=True)
    fw_optimizer = optim.SGD([feature_weights], lr=lr)

    # 损失函数
    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)  # 在这里应用类别权重

    # 训练参数
    epochs = 100
    batch_size = 32
    alpha = 0.1  # 正则化系数

    # 训练循环
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        classifier.train()

        for _ in range(len(X_train) // batch_size):
            # 获取真实数据
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            real_data = X_train[indices]
            real_labels = y_train[indices]

            # 生成假数据
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise, feature_weights)

            # 判别器训练
            d_optimizer.zero_grad()
            real_preds = discriminator(real_data)
            fake_preds = discriminator(fake_data.detach())
            d_loss_real = criterion_bce(real_preds, torch.ones_like(real_preds))
            d_loss_fake = criterion_bce(fake_preds, torch.zeros_like(fake_preds))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 生成器训练
            g_optimizer.zero_grad()
            fake_preds = discriminator(fake_data)
            g_loss = criterion_bce(fake_preds, torch.ones_like(fake_preds))
            g_loss.backward()
            g_optimizer.step()

            # 分类器训练
            c_optimizer.zero_grad()
            real_logits = classifier(real_data)
            fake_logits = classifier(fake_data.detach())
            c_loss_real = criterion_ce(real_logits, real_labels)
            c_loss_fake = criterion_ce(fake_logits, torch.randint(0, num_classes, (batch_size,)))
            c_loss = c_loss_real + c_loss_fake
            c_loss.backward()
            c_optimizer.step()

            # 特征权重更新
            fw_optimizer.zero_grad()
            reg_loss = alpha * torch.norm(feature_weights, p=2)  # 正则化损失
            reg_loss.backward()  # 独立更新特征权重
            fw_optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, C Loss: {c_loss.item():.4f}")

    # 模型评估
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(X_test)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
        test_preds = np.argmax(test_logits.numpy(), axis=1)

    # 计算指标
    auc_score = roc_auc_score(y_test.numpy(), test_probs)
    precision, recall, _ = precision_recall_curve(y_test.numpy(), test_probs)
    pr_auc = auc(recall, precision)
    brier_score = brier_score_loss(y_test.numpy(), test_probs)
    tn, fp, fn, tp = confusion_matrix(y_test.numpy(), test_preds).ravel()
    gmeans = np.sqrt(tp / (tp + fn) * tn / (tn + fp))

    classifier.save(f'models/{data_name}{desired_IR}_{model_name}.h5')
    # 保存结果到文件
    # result_path = os.path.join(results_dir, f"PAKDD_{desired_IR}_DFG_results.txt")
    # with open(result_path, "w") as f:
    #     f.write(f"{auc_score:.4f}\n")
    #     f.write(f"{pr_auc:.4f}\n")
    #     f.write(f"{brier_score:.4f}\n")
    #     f.write(f"{gmeans:.4f}\n")
