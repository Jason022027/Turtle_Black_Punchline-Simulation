import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 設置日誌
def setup_logging():
    """設置日誌配置"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'tblp_cnn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(thickness_file, states_file):
    """載入膜厚和狀態數據"""
    try:
        thickness_df = pd.read_csv(thickness_file)
        states_df = pd.read_csv(states_file)

        thickness_columns = [f'Sample_{i}' for i in range(49)]
        states_columns = [f'Sample_{i}' for i in range(272)]

        X = thickness_df[thickness_columns].values
        y = states_df[states_columns].values

        logging.info(f"膜厚數據維度: {X.shape}")
        logging.info(f"狀態數據維度: {y.shape}")

        return X, y
    except Exception as e:
        logging.error(f"數據載入失敗: {str(e)}")
        raise

def preprocess_data(X, y):
    """預處理數據"""
    try:
        # 數據標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 重塑數據形狀 (100個樣本, 49個特徵, 1個通道)
        X_scaled = X_scaled.reshape(-1, 49, 1)
        
        # 轉換目標變量 (100個樣本, 272個輸出)
        y_list = [y[:, i].astype(np.int32) for i in range(272)]
        
        # 保存標準化器
        np.save('thickness_scaler.npy', scaler)
        
        logging.info(f"預處理後的輸入數據形狀: {X_scaled.shape}")
        logging.info(f"預處理後的輸出數據形狀: {len(y_list)}個輸出，每個輸出形狀: {y_list[0].shape}")
        
        return X_scaled, y_list, scaler
    except Exception as e:
        logging.error(f"數據預處理失敗: {str(e)}")
        raise

def build_cnn_model(input_shape=(49, 1)):
    """建立CNN模型"""
    try:
        inputs = Input(shape=input_shape)
        
        # 第一層卷積
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        # 第二層卷積
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        # 第三層卷積
        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        x = Flatten()(x)
        
        # 全連接層
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 272個輸出層
        outputs = [Dense(3, activation='softmax', name=f'output_{i}')(x) for i in range(272)]
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 編譯模型
        model.compile(
            optimizer='adam',
            loss=['sparse_categorical_crossentropy']*272,
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logging.error(f"模型建立失敗: {str(e)}")
        raise

def train_model(X, y, model, epochs=100, batch_size=32):
    """訓練模型"""
    try:
        # 分割訓練集和驗證集
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 對每個輸出標籤進行分割
        y_train_list = []
        y_val_list = []
        for i in range(len(y)):
            y_train, y_val = train_test_split(y[i], test_size=0.2, random_state=42)
            y_train_list.append(y_train)
            y_val_list.append(y_val)
        
        # 設置回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # 訓練模型
        history = model.fit(
            X_train, y_train_list,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val_list),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    except Exception as e:
        logging.error(f"模型訓練失敗: {str(e)}")
        raise

def plot_training_history(history):
    """繪製訓練歷史"""
    try:
        # 創建圖表目錄
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 繪製損失曲線
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
        plt.close()
        
        # 繪製準確率曲線
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'))
        plt.close()
    except Exception as e:
        logging.error(f"繪製訓練歷史失敗: {str(e)}")

def main():
    """主程序"""
    try:
        # 設置日誌
        setup_logging()
        
        # 設置數據文件路徑
        thickness_file = 'tblp_thickness_wide_test_20250428.csv'
        states_file = 'tblp_states_wide_test_20250428.csv'
        
        # 載入數據
        logging.info("開始載入數據...")
        X, y = load_data(thickness_file, states_file)
        
        # 預處理數據
        logging.info("開始預處理數據...")
        X_scaled, y_list, scaler = preprocess_data(X, y)
        
        # 建立模型
        logging.info("開始建立模型...")
        model = build_cnn_model()
        
        # 訓練模型
        logging.info("開始訓練模型...")
        history = train_model(X_scaled, y_list, model, epochs=100, batch_size=32)
        
        # 繪製訓練歷史
        logging.info("繪製訓練歷史...")
        plot_training_history(history)
        
        # 保存模型 - 使用不同的方式
        try:
            # 方法1：使用 model.save
            model.save('tblp_model_enhanced.h5', save_format='h5')
            logging.info("模型已保存 (方法1): tblp_model_enhanced.h5")
        except Exception as e:
            logging.error(f"方法1保存失敗: {str(e)}")
            try:
                # 方法2：使用 tf.keras.models.save_model
                tf.keras.models.save_model(model, 'tblp_model_enhanced.h5', save_format='h5')
                logging.info("模型已保存 (方法2): tblp_model_enhanced.h5")
            except Exception as e:
                logging.error(f"方法2保存失敗: {str(e)}")
                try:
                    # 方法3：使用 SavedModel 格式
                    tf.keras.models.save_model(model, 'tblp_model_enhanced', save_format='tf')
                    logging.info("模型已保存 (方法3): tblp_model_enhanced")
                except Exception as e:
                    logging.error(f"所有保存方法都失敗: {str(e)}")
                    raise
        
        # 保存標準化器
        np.save('thickness_scaler_enhanced.npy', scaler)
        logging.info("標準化器已保存: thickness_scaler_enhanced.npy")
        
    except Exception as e:
        logging.error(f"程序執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 