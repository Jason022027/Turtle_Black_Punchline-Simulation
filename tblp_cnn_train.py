import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tblp_cnn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_data(thickness_file, states_file):
    """載入膜厚和狀態數據"""
    try:
        thickness_df = pd.read_csv(thickness_file)
        states_df = pd.read_csv(states_file)
        logging.info(f"成功載入數據: {thickness_file} 和 {states_file}")
        return thickness_df, states_df
    except Exception as e:
        logging.error(f"載入數據時發生錯誤: {str(e)}")
        raise

def preprocess_data(thickness_df, states_df):
    """預處理數據"""
    try:
        # 提取49個點位的膜厚值
        thickness_columns = [f'Sample_{i}' for i in range(49)]
        X = thickness_df[thickness_columns].values
        
        # 提取TBLP狀態
        y = states_df['TBLP_State'].values
        
        # 標準化膜厚值
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 重塑數據為CNN輸入格式
        X_reshaped = X_scaled.reshape(-1, 49, 1)
        
        # 保存scaler
        np.save('thickness_scaler.npy', scaler.scale_)
        np.save('thickness_mean.npy', scaler.mean_)
        
        logging.info(f"數據預處理完成，X shape: {X_reshaped.shape}, y shape: {y.shape}")
        return X_reshaped, y, scaler
    except Exception as e:
        logging.error(f"數據預處理時發生錯誤: {str(e)}")
        raise

def build_cnn_model():
    """建立CNN模型"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(49, 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3個類別：關、半開、開
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val):
    """訓練模型"""
    try:
        model = build_cnn_model()
        
        # 設置早停
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 訓練模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 保存模型
        model.save('tblp_cnn_model.h5')
        
        # 繪製訓練曲線
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        logging.info("模型訓練完成並保存")
        return model, history
    except Exception as e:
        logging.error(f"模型訓練時發生錯誤: {str(e)}")
        raise

def main():
    """主函數"""
    try:
        # 載入數據
        thickness_df, states_df = load_data(
            'tblp_thickness_wide_test.csv',
            'tblp_states_wide_test.csv'
        )
        
        # 預處理數據
        X, y, scaler = preprocess_data(thickness_df, states_df)
        
        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 訓練模型
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        # 評估模型
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        logging.info(f"驗證集準確率: {val_accuracy:.4f}")
        
    except Exception as e:
        logging.error(f"程式執行時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 