# TBLP Forward Model 專案

## 專案概述

根據 `tblp_dialogue_complete.txt` 的討論內容，本專案重新設計了 TBLP (Tunable Block Plate) 機器學習系統，專注於學習 **ΔTBLP → Δ膜厚** 的前向關係，並在預測階段整合 SPEC 要求來提供 TBLP 調整建議。

## 核心理念

### 🎯 訓練階段
- **學習目標**: ΔTBLP (272維狀態變化) → Δ膜厚 (49維膜厚變化)
- **模型類型**: Forward Model (前向模型)
- **不包含 SPEC**: 純粹學習物理對應關係，避免過度擬合特定目標

### 🎯 預測階段
- **輸入**: 當前膜厚分布
- **目標**: SPEC (平均值 85.0, 標準差 < 3.0)
- **輸出**: 最佳 TBLP 調整建議

## 檔案結構

```
TBLP_Project/
├── tblp_forward_trainer.py        # 主要訓練程式
├── tblp_spec_optimizer.py         # SPEC 導向優化程式
├── tblp_prediction_interface.py   # 簡單預測介面
├── README_TBLP_Project.md         # 專案說明檔案
└── data/
    ├── tblp_thickness_wide_test_20250428.csv
    └── tblp_states_wide_test_20250428.csv
```

## 使用流程

### 1. 訓練 Forward Model

```bash
python tblp_forward_trainer.py
```

**功能**:
- 載入膜厚和 TBLP 狀態數據
- 建立差分資料 (Δ)
- 訓練深度神經網路模型
- 輸出訓練曲線和模型檔案

**輸出檔案**:
- `tblp_forward_model_YYYYMMDD_HHMMSS.h5` (訓練好的模型)
- `delta_tblp_scaler.npy` (ΔTBLP 標準化器)
- `delta_thickness_scaler.npy` (Δ膜厚 標準化器)
- `logs/tblp_forward_training_*.log` (訓練日誌)
- `plots/forward_training_history.png` (訓練曲線)

### 2. SPEC 導向優化

```bash
python tblp_spec_optimizer.py
```

**功能**:
- 載入訓練好的模型
- 根據當前膜厚和目標 SPEC 找出最佳 TBLP 調整
- 使用數學優化算法搜尋最佳解
- 生成視覺化結果和調整建議

**輸出檔案**:
- `tblp_adjustment_suggestions_YYYYMMDD_HHMMSS.csv` (調整建議)
- `plots/optimization_results_*.png` (優化結果圖)
- `logs/tblp_optimizer_*.log` (優化日誌)

### 3. 簡單預測介面

```bash
python tblp_prediction_interface.py
```

**功能**:
- 互動式膜厚數據輸入
- 基本統計分析和建議
- 模型載入和預測接口

## 核心技術特點

### ✅ 正確的學習方向
- **Forward Model**: ΔTBLP → Δ膜厚
- **物理意義**: 學習 TBLP 調整對膜厚的影響
- **預測應用**: 能真正用於控制建議

### ✅ 分層設計
- **訓練階段**: 純粹學習物理關係
- **預測階段**: 整合業務目標 (SPEC)
- **靈活性**: 可輕易改變目標規格

### ✅ 數學優化
- **目標函數**: 最小化與 SPEC 的差距
- **約束條件**: 限制 TBLP 變化範圍
- **多重初始化**: 提高找到全域最優解的機率

## 模型架構

### Forward Model (MLP)
```
輸入 (272) → Dense(512) → BatchNorm → Dropout(0.3)
           → Dense(256) → BatchNorm → Dropout(0.2)
           → Dense(128) → BatchNorm → Dropout(0.1)
           → Dense(64)  → BatchNorm
           → 輸出 (49)
```

### 優化目標函數
```
Cost = w1 × |平均值 - 85.0| + w2 × max(0, 標準差 - 3.0) + w3 × Σ|ΔTBLP|
```

## 與原始程式的差異

| 項目 | 原始 tblp_cnn_train_enhanced.py | 新版 Forward Model |
|------|--------------------------------|-------------------|
| 學習方向 | 膜厚 → TBLP (反向) | ΔTBLP → Δ膜厚 (前向) |
| 輸入 | 49維膜厚絕對值 | 272維 TBLP 狀態變化 |
| 輸出 | 272個 softmax 分類 | 49維膜厚變化 (連續值) |
| 損失函數 | 分類交叉熵 | 均方誤差 (MSE) |
| 應用場景 | 膜厚反推 TBLP | TBLP 調整預測膜厚 |
| SPEC 整合 | 無 | 在預測階段整合 |

## 使用建議

### 🔧 參數調整
- **權重設定**: 在 `tblp_spec_optimizer.py` 中調整 `weights` 參數
- **TBLP 變化限制**: 調整 `max_change` 參數
- **訓練參數**: 在 `tblp_forward_trainer.py` 中調整 epochs, batch_size

### 📊 結果解讀
- **R² > 0.8**: 模型學習效果良好
- **MAE < 1.0**: 預測誤差在可接受範圍
- **達標狀態**: 優化結果是否滿足 SPEC 要求

## 未來擴展

1. **強化學習**: 可進一步使用 RL 進行動態調整
2. **多目標優化**: 整合更多製程目標
3. **實時調整**: 與實際 TBLP 控制系統整合
4. **不確定性估計**: 加入預測不確定性分析

## 依賴套件

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib scipy
```

## 注意事項

- 確保數據檔案 `tblp_thickness_wide_test_20250428.csv` 和 `tblp_states_wide_test_20250428.csv` 存在
- 訓練完成後才能使用優化程式
- 所有標準化器檔案 (`.npy`) 都必須保留以供預測使用

---

**根據 tblp_dialogue_complete.txt 的設計理念建立，實現了真正符合需求的 TBLP 控制預測系統** 