import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt

def setup_logging():
    """設置日誌配置"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'tblp_forward_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_and_prepare_difference_data(thickness_file, states_file):
    """載入數據並建立差分資料（ΔTBLP → Δ膜厚）"""
    try:
        thickness_df = pd.read_csv(thickness_file)
        states_df = pd.read_csv(states_file)

        thickness_columns = [f'Sample_{i}' for i in range(49)]
        states_columns = [f'Sample_{i}' for i in range(272)]

        thickness_data = thickness_df[thickness_columns].values
        states_data = states_df[states_columns].values

        logging.info(f"原始膜厚數據維度: {thickness_data.shape}")
        logging.info(f"原始狀態數據維度: {states_data.shape}")

        delta_tblp = states_data[1:] - states_data[:-1]
        delta_thickness = thickness_data[1:] - thickness_data[:-1]

        logging.info(f"ΔTBLP 數據維度: {delta_tblp.shape}")
        logging.info(f"Δ膜厚 數據維度: {delta_thickness.shape}")
        
        return delta_tblp, delta_thickness

    except Exception as e:
        logging.error(f"數據載入失敗: {str(e)}")
        raise

def preprocess_data(X, y):
    """預處理差分數據"""
    try:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        np.save('delta_tblp_scaler.npy', scaler_X)
        np.save('delta_thickness_scaler.npy', scaler_y)
        
        logging.info(f"預處理後的 ΔTBLP 形狀: {X_scaled.shape}")
        logging.info(f"預處理後的 Δ膜厚 形狀: {y_scaled.shape}")
        
        return X_scaled, y_scaled, scaler_X, scaler_y
    
    except Exception as e:
        logging.error(f"數據預處理失敗: {str(e)}")
        raise

def build_forward_model(input_dim=272, output_dim=49):
    """建立 Forward 模型：ΔTBLP → Δ膜厚"""
    try:
        inputs = Input(shape=(input_dim,), name='delta_tblp_input')
        
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        outputs = Dense(output_dim, activation='linear', name='delta_thickness_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logging.info("Forward 模型建立成功")
        return model
    
    except Exception as e:
        logging.error(f"模型建立失敗: {str(e)}")
        raise

def train_forward_model(X, y, model, epochs=200, batch_size=16):
    """訓練 Forward 模型"""
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history, (X_train, X_val, y_train, y_val)
    
    except Exception as e:
        logging.error(f"模型訓練失敗: {str(e)}")
        raise

def evaluate_model(model, X_val, y_val, scaler_y):
    """評估模型性能"""
    try:
        y_pred = model.predict(X_val)
        
        y_val_original = scaler_y.inverse_transform(y_val)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        
        mse = mean_squared_error(y_val_original, y_pred_original)
        mae = mean_absolute_error(y_val_original, y_pred_original)
        r2 = r2_score(y_val_original, y_pred_original)
        
        logging.info(f"模型評估結果:")
        logging.info(f"MSE: {mse:.6f}")
        logging.info(f"MAE: {mae:.6f}")
        logging.info(f"R²: {r2:.6f}")
        
        return {'mse': mse, 'mae': mae, 'r2': r2}
    
    except Exception as e:
        logging.error(f"模型評估失敗: {str(e)}")
        raise

def plot_training_history(history):
    """繪製訓練歷史"""
    try:
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'forward_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("訓練歷史圖已保存")
        
    except Exception as e:
        logging.error(f"繪製訓練歷史失敗: {str(e)}")

def main():
    """主程序"""
    try:
        setup_logging()
        logging.info("=" * 60)
        logging.info("TBLP Forward Model Training Started")
        logging.info("目標：學習 ΔTBLP → Δ膜厚 的關係")
        logging.info("=" * 60)
        
        thickness_file = 'tblp_thickness_wide_test_20250428.csv'
        states_file = 'tblp_states_wide_test_20250428.csv'
        
        logging.info("步驟 1: 載入並建立差分數據...")
        delta_tblp, delta_thickness = load_and_prepare_difference_data(thickness_file, states_file)
        
        logging.info("步驟 2: 預處理數據...")
        X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(delta_tblp, delta_thickness)
        
        logging.info("步驟 3: 建立 Forward 模型...")
        model = build_forward_model()
        
        logging.info("步驟 4: 訓練模型...")
        history, (X_train, X_val, y_train, y_val) = train_forward_model(
            X_scaled, y_scaled, model, epochs=200, batch_size=16
        )
        
        logging.info("步驟 5: 評估模型...")
        eval_results = evaluate_model(model, X_val, y_val, scaler_y)
        
        logging.info("步驟 6: 繪製訓練結果...")
        plot_training_history(history)
        
        logging.info("步驟 7: 保存模型...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'tblp_forward_model_{timestamp}.h5'
        model.save(model_name)
        logging.info(f"模型已保存: {model_name}")
        
        logging.info("=" * 60)
        logging.info("訓練完成！")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"程序執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 