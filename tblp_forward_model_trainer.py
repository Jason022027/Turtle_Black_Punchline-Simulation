import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        # 載入原始數據
        thickness_df = pd.read_csv(thickness_file)
        states_df = pd.read_csv(states_file)

        thickness_columns = [f'Sample_{i}' for i in range(49)]
        states_columns = [f'Sample_{i}' for i in range(272)]

        thickness_data = thickness_df[thickness_columns].values
        states_data = states_df[states_columns].values

        logging.info(f"原始膜厚數據維度: {thickness_data.shape}")
        logging.info(f"原始狀態數據維度: {states_data.shape}")

        # 計算差分（前後組的變化）
        # ΔTBLP = states[t] - states[t-1]  (輸入)
        # Δ膜厚 = thickness[t] - thickness[t-1]  (輸出目標)
        delta_tblp = states_data[1:] - states_data[:-1]  # (n-1, 272)
        delta_thickness = thickness_data[1:] - thickness_data[:-1]  # (n-1, 49)

        logging.info(f"ΔTBLP 數據維度: {delta_tblp.shape}")
        logging.info(f"Δ膜厚 數據維度: {delta_thickness.shape}")
        
        # 檢查數據統計
        logging.info(f"ΔTBLP 範圍: [{delta_tblp.min():.3f}, {delta_tblp.max():.3f}]")
        logging.info(f"Δ膜厚 範圍: [{delta_thickness.min():.3f}, {delta_thickness.max():.3f}]")
        
        return delta_tblp, delta_thickness

    except Exception as e:
        logging.error(f"數據載入失敗: {str(e)}")
        raise

def preprocess_data(X, y):
    """預處理差分數據"""
    try:
        # 標準化 ΔTBLP (輸入)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # 標準化 Δ膜厚 (輸出)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        # 保存標準化器
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
        
        # 深度全連接網路
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
        
        # 輸出層 - 49維膜厚變化（連續值）
        outputs = Dense(output_dim, activation='linear', name='delta_thickness_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 編譯模型 - 使用回歸損失函數
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logging.info("Forward 模型建立成功")
        logging.info(f"模型摘要:")
        model.summary(print_fn=logging.info)
        
        return model
    
    except Exception as e:
        logging.error(f"模型建立失敗: {str(e)}")
        raise

def build_cnn_forward_model(input_dim=272, output_dim=49):
    """建立 CNN Forward 模型：ΔTBLP → Δ膜厚"""
    try:
        inputs = Input(shape=(input_dim, 1), name='delta_tblp_input')
        
        # CNN 層
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
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
        
        # 輸出層 - 49維膜厚變化（連續值）
        outputs = Dense(output_dim, activation='linear', name='delta_thickness_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 編譯模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logging.info("CNN Forward 模型建立成功")
        logging.info(f"模型摘要:")
        model.summary(print_fn=logging.info)
        
        return model
    
    except Exception as e:
        logging.error(f"CNN 模型建立失敗: {str(e)}")
        raise

def train_forward_model(X, y, model, epochs=200, batch_size=16, use_cnn=False):
    """訓練 Forward 模型"""
    try:
        # 如果使用 CNN，需要重塑輸入
        if use_cnn:
            X = X.reshape(-1, X.shape[1], 1)
            logging.info(f"CNN 輸入形狀: {X.shape}")
        
        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logging.info(f"訓練集大小: {X_train.shape[0]}")
        logging.info(f"驗證集大小: {X_val.shape[0]}")
        
        # 設置回調函數
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
        
        # 訓練模型
        logging.info("開始訓練模型...")
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
        # 預測
        y_pred = model.predict(X_val)
        
        # 反標準化
        y_val_original = scaler_y.inverse_transform(y_val)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        
        # 計算指標
        mse = mean_squared_error(y_val_original, y_pred_original)
        mae = mean_absolute_error(y_val_original, y_pred_original)
        r2 = r2_score(y_val_original, y_pred_original)
        
        logging.info(f"模型評估結果:")
        logging.info(f"MSE: {mse:.6f}")
        logging.info(f"MAE: {mae:.6f}")
        logging.info(f"R²: {r2:.6f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_true': y_val_original,
            'y_pred': y_pred_original
        }
    
    except Exception as e:
        logging.error(f"模型評估失敗: {str(e)}")
        raise

def plot_training_history(history):
    """繪製訓練歷史"""
    try:
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 繪製損失曲線
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 繪製 MAE
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # 繪製學習率（如果有）
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'forward_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("訓練歷史圖已保存到 plots/forward_training_history.png")
        
    except Exception as e:
        logging.error(f"繪製訓練歷史失敗: {str(e)}")

def plot_prediction_results(eval_results):
    """繪製預測結果分析"""
    try:
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        y_true = eval_results['y_true']
        y_pred = eval_results['y_pred']
        
        # 繪製預測 vs 真實值
        plt.figure(figsize=(15, 10))
        
        # 整體預測散點圖
        plt.subplot(2, 3, 1)
        plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs True Values\nR² = {eval_results["r2"]:.4f}')
        plt.grid(True)
        
        # 誤差分佈
        plt.subplot(2, 3, 2)
        errors = (y_pred - y_true).flatten()
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        
        # 前幾個樣本的預測對比
        plt.subplot(2, 3, 3)
        sample_idx = 0
        plt.plot(y_true[sample_idx], 'o-', label='True', alpha=0.7)
        plt.plot(y_pred[sample_idx], 's-', label='Predicted', alpha=0.7)
        plt.xlabel('膜厚點位置')
        plt.ylabel('Δ膜厚值')
        plt.title(f'Sample {sample_idx} Prediction')
        plt.legend()
        plt.grid(True)
        
        # 各點位的預測準確度
        plt.subplot(2, 3, 4)
        point_mae = np.mean(np.abs(y_pred - y_true), axis=0)
        plt.bar(range(len(point_mae)), point_mae)
        plt.xlabel('膜厚點位置')
        plt.ylabel('MAE')
        plt.title('各點位預測誤差')
        plt.grid(True)
        
        # 預測值範圍比較
        plt.subplot(2, 3, 5)
        true_range = np.max(y_true, axis=1) - np.min(y_true, axis=1)
        pred_range = np.max(y_pred, axis=1) - np.min(y_pred, axis=1)
        plt.scatter(true_range, pred_range, alpha=0.7)
        plt.plot([true_range.min(), true_range.max()], [true_range.min(), true_range.max()], 'r--', lw=2)
        plt.xlabel('True Range')
        plt.ylabel('Predicted Range')
        plt.title('膜厚變化範圍預測')
        plt.grid(True)
        
        # 累積誤差分析
        plt.subplot(2, 3, 6)
        cumulative_error = np.cumsum(np.abs(errors))
        plt.plot(cumulative_error)
        plt.xlabel('樣本索引')
        plt.ylabel('累積絕對誤差')
        plt.title('累積預測誤差')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("預測結果分析圖已保存到 plots/prediction_analysis.png")
        
    except Exception as e:
        logging.error(f"繪製預測結果失敗: {str(e)}")

def main():
    """主程序"""
    try:
        # 設置日誌
        setup_logging()
        logging.info("=" * 60)
        logging.info("TBLP Forward Model Training Started")
        logging.info("目標：學習 ΔTBLP → Δ膜厚 的關係")
        logging.info("=" * 60)
        
        # 設置數據文件路徑
        thickness_file = 'tblp_thickness_wide_test_20250428.csv'
        states_file = 'tblp_states_wide_test_20250428.csv'
        
        # 載入並準備差分數據
        logging.info("步驟 1: 載入並建立差分數據...")
        delta_tblp, delta_thickness = load_and_prepare_difference_data(thickness_file, states_file)
        
        # 預處理數據
        logging.info("步驟 2: 預處理數據...")
        X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(delta_tblp, delta_thickness)
        
        # 選擇模型類型 (可以改為 True 使用 CNN)
        use_cnn_model = False
        
        # 建立模型
        logging.info(f"步驟 3: 建立 {'CNN' if use_cnn_model else 'MLP'} Forward 模型...")
        if use_cnn_model:
            model = build_cnn_forward_model()
        else:
            model = build_forward_model()
        
        # 訓練模型
        logging.info("步驟 4: 訓練模型...")
        history, (X_train, X_val, y_train, y_val) = train_forward_model(
            X_scaled, y_scaled, model, 
            epochs=200, 
            batch_size=16,
            use_cnn=use_cnn_model
        )
        
        # 評估模型
        logging.info("步驟 5: 評估模型...")
        eval_results = evaluate_model(model, X_val, y_val, scaler_y)
        
        # 繪製結果
        logging.info("步驟 6: 繪製訓練和預測結果...")
        plot_training_history(history)
        plot_prediction_results(eval_results)
        
        # 保存模型和標準化器
        logging.info("步驟 7: 保存模型和標準化器...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_name = f'tblp_forward_model_{timestamp}.h5'
        model.save(model_name)
        logging.info(f"模型已保存: {model_name}")
        
        # 保存最終結果摘要
        summary = {
            'model_type': 'CNN' if use_cnn_model else 'MLP',
            'input_dim': X_scaled.shape[1],
            'output_dim': y_scaled.shape[1],
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'final_mse': eval_results['mse'],
            'final_mae': eval_results['mae'],
            'final_r2': eval_results['r2'],
            'timestamp': timestamp
        }
        
        logging.info("=" * 60)
        logging.info("訓練完成！模型摘要:")
        for key, value in summary.items():
            logging.info(f"{key}: {value}")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"程序執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 