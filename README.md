# TBLP膜厚預測系統

這是一個用於預測TBLP（半導體製程機台組件）狀態的深度學習系統。系統可以根據49個點位的膜厚測量值，預測TBLP的狀態（開、半開、關）。

## 功能特點

- 使用CNN模型處理49個點位的空間數據
- 自動數據預處理和標準化
- 完整的訓練和預測流程
- 詳細的日誌記錄
- 可視化訓練過程
- 支持批量預測

## 系統要求

- Python 3.8+
- 見 requirements.txt 中的依賴套件

## 安裝

1. 克隆專案：
```bash
git clone [專案URL]
cd [專案目錄]
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成示例數據
```bash
python generate_sample_data.py
```
這將生成：
- 訓練數據（膜厚和狀態）
- 新的預測數據

### 2. 訓練模型
```bash
python tblp_cnn_train.py
```
這將：
- 訓練CNN模型
- 保存模型和scaler
- 生成訓練過程圖表

### 3. 進行預測
```bash
python tblp_cnn_predict.py
```
這將：
- 加載訓練好的模型
- 讀取新的膜厚數據
- 預測TBLP狀態
- 保存預測結果

## 文件說明

- `tblp_cnn_train.py`: 模型訓練程式
- `tblp_cnn_predict.py`: 預測程式
- `generate_sample_data.py`: 數據生成程式
- `requirements.txt`: 依賴套件列表

## 數據格式

### 輸入數據格式
CSV文件應包含以下列：
- `number`: 編號
- `Sample_0` 到 `Sample_48`: 49個點位的膜厚值

### 輸出數據格式
預測結果將包含：
- 原始數據列
- `Predicted_State`: 預測的TBLP狀態（0:關, 1:半開, 2:開）
- `State_Description`: 狀態描述

## 注意事項

1. 確保輸入數據包含49個點位的膜厚值
2. 膜厚值應在合理範圍內（約75-90nm）
3. 建議使用標準化的數據進行訓練和預測

## 日誌文件

系統會自動生成以下日誌文件：
- 訓練日誌：`tblp_cnn_training_YYYYMMDD_HHMMSS.log`
- 預測日誌：`tblp_cnn_prediction_YYYYMMDD_HHMMSS.log`
- 數據生成日誌：`sample_data_generation_YYYYMMDD_HHMMSS.log`

## 聯繫方式

如有問題，請聯繫：[聯繫方式] 