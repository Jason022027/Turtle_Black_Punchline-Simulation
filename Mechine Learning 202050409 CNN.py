import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: 產生模擬資料
# ----------------------------
n_samples = 1000           # 樣本數
n_tblp = 272               # TBLP 控制點數量
n_thickness_points = 49    # Wafer 膜厚測量點數

# 產生「調整前」的 TBLP 狀態，隨機取 0, 1, 2
TBLP_before = np.random.randint(0, 3, size=(n_samples, n_tblp))

# 隨機產生一個改變，每個 TBLP 控制點可能變化 -1、0 或 +1，
# 注意結果必須介於 0 與 2 之間
delta_tblp = np.random.randint(-1, 2, size=(n_samples, n_tblp))
TBLP_after = np.clip(TBLP_before + delta_tblp, 0, 2)

# ----------------------------
# Step 2: 模擬膜厚數據
# ----------------------------
# 假設膜厚跟 TBLP 狀態的關係可用一個線性模型來模擬，
# 這裡使用一個隨機權重矩陣 W（形狀：(272, 49)），再加上一些雜訊
np.random.seed(42)
W = np.random.randn(n_tblp, n_thickness_points)
noise_level = 0.1

# 計算調整前後的膜厚（每一筆資料產生一個 49 維向量）
thickness_before = TBLP_before.dot(W) + noise_level * np.random.randn(n_samples, n_thickness_points)
thickness_after = TBLP_after.dot(W) + noise_level * np.random.randn(n_samples, n_thickness_points)

# ----------------------------
# Step 3: 計算差分數據
# ----------------------------
# ΔTBLP: 調整後 TBLP 減去 調整前 TBLP，值域介於 -2 至 +2
delta_TBLP = TBLP_after - TBLP_before

# Δ膜厚: 調整後膜厚減去調整前膜厚
delta_thickness = thickness_after - thickness_before

# ----------------------------
# Step 4: 分割訓練與測試數據集
# ----------------------------
X = delta_TBLP  # 輸入特徵：ΔTBLP（272 維）
y = delta_thickness  # 輸出標籤：Δ膜厚（49 維）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 5: 建立並訓練模型
# ----------------------------
# 這裡我們使用一個簡單的多層感知機 (MLP) 來做回歸任務
model = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Step 6: 模型評估
# ----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("測試資料 MSE:", mse)

# 畫出部分樣本的預測 vs 實際 Δ膜厚（以第一個測量點為例）
plt.figure(figsize=(8, 5))
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
plt.xlabel("實際 Δ膜厚（第一點）")
plt.ylabel("預測 Δ膜厚（第一點）")
plt.title("Δ膜厚 預測 vs 實際")
plt.show()

# ----------------------------
# Step 7: 範例使用：給定一組 TBLP 調整，預測其膜厚變化
# ----------------------------
# 假設我們想測試只調整第一個 TBLP 控制點（改變 +1），其他不變：
sample_delta_TBLP = np.zeros((1, n_tblp))
sample_delta_TBLP[0, 0] = 1  # 只改變第一個控制點

predicted_delta_thickness = model.predict(sample_delta_TBLP)
print("範例輸入 ΔTBLP:", sample_delta_TBLP)
print("預測的 Δ膜厚:", predicted_delta_thickness)
