# -*- coding: utf-8 -*-
"""
一键运行：比较 AAE 与 DAE 生成样本到 DNN 决策边界的平均距离
- 读取 ImbalanceData/{data_name}_{desired_IR}.txt
- 训练 DNN + AAE（你的 AT 逻辑）
- 训练 DAE 作为对照
- 计算两种“到决策边界的距离”：一阶近似 与 沿梯度方向二分搜索
- 导出 CSV：summary 与 detail
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

# ========== 可改参数 ==========
data_name = 'prosper'
desired_IR = 10
random_seed = 0
t_max_search = 0.5  # 二分搜索最大半径（特征在[0,1]时0.5通常足够）
# ============================

os.makedirs("models", exist_ok=True)

# ========== 评价指标 ==========
def cal_result(Cls, y_te, X_te):
    L = []
    y_prob = Cls.predict(X_te, verbose=0)  # [N,2]
    ROC = roc_auc_score(y_te, y_prob[:, 1])
    PR = average_precision_score(y_te, y_prob[:, 1])
    BS = brier_score_loss(y_te, y_prob[:, 1])
    fpr, tpr, _ = roc_curve(y_te, y_prob[:, 1])
    G_means = np.sqrt(tpr * (1 - fpr)).max()
    L.extend([ROC, PR, BS, G_means])
    return np.array(L)

# ========== 损失函数 ==========
def TNNLS_loss(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    n = -tf.reduce_mean(tf.math.log(1. - pt_0 + tf.keras.backend.epsilon()))
    p = -tf.reduce_mean(tf.math.log(pt_1+ tf.keras.backend.epsilon()))
    z = -tf.reduce_mean(y_pred * (1-y_pred))
    return n + 2*p + z

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# ========== 辅助函数（边界距离） ==========
def project_to_box(X, low=0.0, high=1.0):
    return np.clip(X, low, high)

def prob_and_grad(model, X, batch_size=1024):
    """
    返回 p(x)=softmax输出[:,1] 以及 ∇_x p(x)
    """
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    ps, grads = [], []
    for i in range(0, X_tf.shape[0], batch_size):
        xb = X_tf[i:i+batch_size]
        with tf.GradientTape() as tape:
            tape.watch(xb)
            probs = model(xb, training=False)  # [B,2]
            p1 = probs[:, 1]
        grad = tape.gradient(p1, xb)          # [B,d]
        ps.append(p1.numpy())
        grads.append(grad.numpy())
    return np.concatenate(ps, axis=0), np.concatenate(grads, axis=0)

def grad_norm_distance_estimate(model, X, eps=1e-12, batch_size=1024):
    """
    一阶近似距离：|p-0.5| / ||∇p||_2
    """
    p, g = prob_and_grad(model, X, batch_size=batch_size)
    gnorm = np.linalg.norm(g, axis=1) + eps
    d_hat = np.abs(p - 0.5) / gnorm
    return d_hat, p, g

def boundary_distance_bisect(model, X, p=None, g=None,
                             t_max=0.5, steps=20, tol=1e-5, box=(0.0, 1.0)):
    """
    沿梯度方向做二分线搜索，求使 p(x_t)=0.5 的最小 t。
    - 返回每个样本的距离（一维)
    - 未命中（找不到跨越0.5）时返回 t_max 作为下界
    """
    import numpy as np

    # 需要 p 与 ∇p
    if p is None or g is None:
        p, g = prob_and_grad(model, X)  # p: [N], g: [N,d]

    N, d = X.shape
    X0 = X.astype(np.float32)

    # 单位方向：p>=0.5 往 -grad，p<0.5 往 +grad
    norms = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    u = g / norms                                  # [N,d]
    signs = np.where(p >= 0.5, -1.0, 1.0).reshape(-1, 1)
    u = signs * u                                   # [N,d]

    # 一维 lo/hi/d，更不容易出错
    lo = np.zeros(N, dtype=np.float32)
    hi = np.full(N, t_max, dtype=np.float32)
    d_out = np.full(N, np.nan, dtype=np.float32)

    # 先检查 hi 是否跨越边界
    X_hi = X0 + hi[:, None] * u
    if box is not None:
        X_hi = np.clip(X_hi, box[0], box[1])
    p_hi, _ = prob_and_grad(model, X_hi)
    hit = ((p - 0.5) * (p_hi - 0.5) <= 0)          # 一维布尔

    # 可选：对未命中的样本再扩一次半径
    if np.any(~hit):
        hi2 = np.where(hit, hi, 2.0 * t_max).astype(np.float32)
        X_hi2 = X0 + hi2[:, None] * u
        if box is not None:
            X_hi2 = np.clip(X_hi2, box[0], box[1])
        p_hi2, _ = prob_and_grad(model, X_hi2)
        hit2 = ((p - 0.5) * (p_hi2 - 0.5) <= 0)
        # 用能命中的 hi2 替换
        need = (~hit) & hit2
        hi[need] = hi2[need]
        hit = hit | hit2

    # 只对能命中的样本做二分
    idx = np.where(hit)[0]
    for _ in range(steps):
        if idx.size == 0:
            break
        mid = (lo[idx] + hi[idx]) / 2.0                       # 一维
        X_mid = X0[idx] + mid[:, None] * u[idx]               # [k,d]
        if box is not None:
            X_mid = np.clip(X_mid, box[0], box[1])
        p_mid, _ = prob_and_grad(model, X_mid)                # [k]

        # 判断哪一侧，保持一维布尔
        left_side = ((p[idx] - 0.5) * (p_mid - 0.5) <= 0)     # [k] bool
        # 收缩区间
        hi[idx[left_side]] = mid[left_side]
        lo[idx[~left_side]] = mid[~left_side]

        # 收敛判据
        done = (hi[idx] - lo[idx]) < tol                      # [k] bool
        if np.any(done):
            done_idx = idx[done]
            d_out[done_idx] = hi[done_idx]
            idx = idx[~done]                                  # 保持一维

    # 未收敛但能命中的，取 hi 近似
    if idx.size > 0:
        d_out[idx] = hi[idx]

    # 未命中的返回 t_max 作为下界
    d_out[~hit] = t_max

    return d_out  # shape [N]

# ========== AT（含 DNN 与 AAE） ==========
class adv_training():
    def __init__(self, X_shape, attacking_strength=0.01, encoding_dim=64, num_iter=100):
        self.attacking_strength = attacking_strength
        self.num_arr = X_shape[1]
        self.encoding_dim = encoding_dim
        self.num_iter = num_iter
        self.batch_size = 32
        self.target = self.build_target()
        self.AE = self.build_autoencoder()
        self.best = np.array([0,0,0,0])
        self.ori = None

    def build_target(self):
        model = Sequential()
        model.add(layers.Dense(64, input_shape=(self.num_arr,), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))
        model.compile(loss=TNNLS_loss, optimizer='adam', metrics=['accuracy'])
        return model

    def build_autoencoder(self):
        inp = Input(shape=(self.num_arr,))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dense(self.encoding_dim, activation='relu')(x)
        x = layers.Dense(self.encoding_dim, activation='relu')(x)
        out = layers.Dense(self.num_arr, activation='sigmoid')(x)
        AE = Model(inp, out)

        def custom_loss(y_true, y_pred):
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(y_pred - y_true), axis=1)
            pos = self.target(y_pred)[:, 1]
            adversarial_loss = self.attacking_strength * pos
            return reconstruction_loss + adversarial_loss

        AE.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss)
        return AE

    def train(self, X_tr, y_tr, X_te, y_te):
        y_tr_cat = to_categorical(y_tr)

        X_p = X_tr[y_tr==1]
        y_p = y_tr_cat[y_tr==1]
        # X_n = X_tr[y_tr==0]  # 如需使用

        # 训练 DNN 初始
        self.target.fit(X_tr, y_tr_cat, epochs=5, batch_size=32, shuffle=True, class_weight={0:1, 1:2}, verbose=0)
        self.ori = cal_result(self.target, y_te, X_te)

        # 训练 AAE 初始（用带微噪声的少数类）
        noise = 0.005 * np.random.normal(size=X_p.shape)
        noisy_p = np.clip(X_p + noise, 0.0, 1.0)
        self.AE.fit(noisy_p, X_p, epochs=100, batch_size=32, shuffle=True, verbose=0)

        for epoch in range(self.num_iter):
            # 生成对抗少数类
            adv = self.AE.predict(X_p + np.random.normal(0, 0.05, X_p.shape), verbose=0)
            adv = np.clip(adv, 0.0, 1.0)
            y_adv = to_categorical(np.ones(len(adv)))

            # 抽取子集 + 拼接生成样本
            subset_ratio = 0.2
            X_batch, _, y_batch, _ = train_test_split(X_tr, y_tr_cat, test_size=(1 - subset_ratio), stratify=y_tr)
            # 取固定数量的对抗样本
            take = min( max(20, len(adv)//10), len(adv) )
            ix_a = np.random.randint(0, len(adv), take)
            X_adv_batch = adv[ix_a]
            y_adv_batch = y_adv[ix_a]
            X_batch = np.vstack((X_batch, X_adv_batch))
            y_batch = np.vstack((y_batch, y_adv_batch))

            lr_scheduler = LearningRateScheduler(scheduler)
            self.target.fit(X_batch, y_batch, shuffle=True, verbose=0, callbacks=[lr_scheduler], class_weight={0:1, 1:1})

            # 更新 AAE
            self.AE.fit(noisy_p, X_p, shuffle=True, verbose=0)

            # 测试记录
            result = cal_result(self.target, y_te, X_te)
            if (result[1] > self.best[1]) and (result[0] > self.best[0]) and (result[3] > self.best[3]):
                self.best = result

            if (epoch+1) % 10 == 0:
                print(f"[AT] epoch {epoch+1}/{self.num_iter} | AUC-ROC={result[0]:.4f} AUC-PR={result[1]:.4f} G-mean={result[3]:.4f}")

        return self.target

# ========== DAE ==========
def build_dae_like(input_dim, encoding_dim=64, lr=0.01):
    inp = Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    out = layers.Dense(input_dim, activation='sigmoid')(x)
    DAE = Model(inp, out)
    DAE.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return DAE

# ========== 主程序 ==========
def main():
    # 读取数据
    path = f'ImbalanceData/{data_name}_{desired_IR}.txt'
    if not os.path.exists(path):
        raise FileNotFoundError(f"未发现数据文件：{path}")
    Data = np.loadtxt(path)
    X, y = Data[:, :-1], Data[:, -1].astype(int)

    # 归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    for train_index, test_index in sss.split(X, y):
        X_tr = X[train_index]
        y_tr = y[train_index]
        X_te = X[test_index]
        y_te = y[test_index]

    print("训练集少数类数量：", int(np.sum(y_tr)))

    # 训练 AT（DNN+AAE）
    AT = adv_training(X_tr.shape, attacking_strength=0.001, encoding_dim=32, num_iter=100)
    model = AT.train(X_tr, y_tr, X_te, y_te)
    model.save(f'models/{data_name}{desired_IR}_AT_target.h5')

    # 少数类与生成样本
    X_p = X_tr[y_tr == 1]
    y_p_pred = model.predict(X_p).argmax(axis=1)

    # 只考虑被DNN分类正确的正类样本
    X_p_misclassified = X_p[y_p_pred != 0]

    if len(X_p_misclassified) == 0:
        print("没有被DNN分类错误的正类样本，程序退出。")
        return

    # AAE 样本
    adv_gen = AT.AE.predict(X_p_misclassified + np.random.normal(0, 0.05, X_p_misclassified.shape), verbose=0)
    adv_gen = np.clip(adv_gen, 0.0, 1.0)

    # 训练 DAE 并生成样本
    DAE = build_dae_like(input_dim=X_tr.shape[1], encoding_dim=64, lr=0.01)
    noise = 0.005 * np.random.normal(size=X_p_misclassified.shape)
    noisy_p = np.clip(X_p_misclassified + noise, 0.0, 1.0)
    DAE.fit(noisy_p, X_p_misclassified, epochs=50, batch_size=64, verbose=0)
    dae_gen = np.clip(DAE.predict(X_p_misclassified, verbose=0), 0.0, 1.0)

    # 计算两种距离
    dhat_orig, p_orig, _ = grad_norm_distance_estimate(model, X_p_misclassified)
    dhat_aae,  p_aae,  _ = grad_norm_distance_estimate(model, adv_gen)
    dhat_dae,  p_dae,  _ = grad_norm_distance_estimate(model, dae_gen)

    d_bisec_orig = boundary_distance_bisect(model, X_p_misclassified, t_max=t_max_search, steps=20, box=(0.0,1.0))
    d_bisec_aae  = boundary_distance_bisect(model, adv_gen,t_max=t_max_search, steps=20, box=(0.0,1.0))
    d_bisec_dae  = boundary_distance_bisect(model, dae_gen,t_max=t_max_search, steps=20, box=(0.0,1.0))

    # 汇总表
    def summarize(name, d_hat, d_bisec, p):
        return {
            "set": name,
            "mean_d_hat":  float(np.mean(d_hat)),
            "median_d_hat":float(np.median(d_hat)),
            "mean_d_bisec":float(np.mean(d_bisec)),
            "median_d_bisec":float(np.median(d_bisec)),
            "mean_p":       float(np.mean(p))
        }

    summary = [
        summarize("orig", dhat_orig, d_bisec_orig, p_orig),
        summarize("AAE",  dhat_aae,  d_bisec_aae,  p_aae),
        summarize("DAE",  dhat_dae,  d_bisec_dae,  p_dae),
    ]
    df_summary = pd.DataFrame(summary)
    print("\n=== 到决策边界的距离对比（平均/中位数） ===")
    print(df_summary.to_string(index=False))

    # 明细表
    df_detail = pd.DataFrame({
        "set": (["orig"]*len(X_p_misclassified)) + (["AAE"]*len(adv_gen)) + (["DAE"]*len(dae_gen)),
        "d_hat": np.concatenate([dhat_orig, dhat_aae, dhat_dae]),
        "d_bisec": np.concatenate([d_bisec_orig, d_bisec_aae, d_bisec_dae]),
        "p": np.concatenate([p_orig, p_aae, p_dae]),
    })

    csv_summary = f"boundary_distance_summary_{data_name}_{desired_IR}.csv"
    csv_detail  = f"boundary_distance_detail_{data_name}_{desired_IR}.csv"
    df_summary.to_csv(csv_summary, index=False)
    df_detail.to_csv(csv_detail, index=False)
    print(f"\n已保存汇总: {csv_summary}")
    print(f"已保存明细: {csv_detail}")

    # 友好打印
    print("\n[Gradient-norm estimate] mean distance to boundary")
    print("  minority originals: {:.6f}".format(np.mean(dhat_orig)))
    print("  AAE generated    : {:.6f}".format(np.mean(dhat_aae)))
    print("  DAE generated    : {:.6f}".format(np.mean(dhat_dae)))

    print("\n[Bisection search] mean distance to boundary")
    print("  minority originals: {:.6f}".format(np.mean(d_bisec_orig)))
    print("  AAE generated    : {:.6f}".format(np.mean(d_bisec_aae)))
    print("  DAE generated    : {:.6f}".format(np.mean(d_bisec_dae)))

if __name__ == "__main__":
    main()