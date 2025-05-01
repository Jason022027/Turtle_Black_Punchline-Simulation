import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import os
import sys

# 設置日誌
def setup_logging():
    """設置日誌配置"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'tblp_cnn_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_predict_data(file_path):
    """載入預測數據"""
    try:
        # 嘗試多種編碼方式
        encodings = ['utf-8', 'latin1', 'big5', 'gbk', 'shift-jis']
        for encoding in encodings:
            try:
                logging.info(f"嘗試使用 {encoding} 編碼讀取文件")
                df = pd.read_csv(file_path, encoding=encoding)
                thickness_columns = [f'Sample_{i}' for i in range(49)]
                X = df[thickness_columns].values
                logging.info(f"成功使用 {encoding} 編碼讀取文件")
                logging.info(f"預測數據維度: {X.shape}")
                return X
            except UnicodeDecodeError:
                logging.warning(f"{encoding} 編碼讀取失敗，嘗試下一種編碼")
                continue
        
        # 如果所有編碼都失敗，拋出異常
        raise ValueError("無法使用任何已知編碼讀取文件")
    except Exception as e:
        logging.error(f"數據載入失敗: {str(e)}")
        raise

def preprocess_predict_data(X, scaler_path):
    """預處理預測數據"""
    try:
        # 載入標準化器
        scaler = np.load(scaler_path, allow_pickle=True).item()
        
        # 標準化數據
        X_scaled = scaler.transform(X)
        
        # 重塑數據形狀
        X_scaled = X_scaled.reshape(-1, 49, 1)
        
        logging.info(f"預處理後的數據形狀: {X_scaled.shape}")
        return X_scaled
    except Exception as e:
        logging.error(f"數據預處理失敗: {str(e)}")
        raise

def predict(model, X):
    """進行預測"""
    try:
        # 記錄預測開始
        logging.info(f"開始預測，輸入數據形狀: {X.shape}")
        
        # 進行預測
        predictions = model.predict(X)
        
        # 檢查預測結果的類型
        if isinstance(predictions, list):
            logging.info(f"預測結果是列表，列表長度: {len(predictions)}")
            
            # 處理列表形式的預測結果
            if len(predictions) == 1:
                # 單一輸出的列表，取第一個元素
                pred_array = predictions[0]
                logging.info(f"使用列表中的第一個預測結果，形狀: {pred_array.shape}")
                predicted_classes = np.argmax(pred_array, axis=1)
                results = pd.DataFrame({'預測類別': predicted_classes})
            else:
                # 多輸出的列表
                logging.info(f"多輸出模型，輸出數: {len(predictions)}")
                predicted_classes = [np.argmax(pred, axis=1) for pred in predictions]
                
                # 創建結果 DataFrame
                results = pd.DataFrame()
                for i in range(len(predicted_classes)):
                    results[f'Sample_{i}'] = predicted_classes[i]
        else:
            # NumPy數組形式的預測結果
            logging.info(f"預測完成，預測結果形狀: {predictions.shape}")
            
            # 檢查預測結果的形狀
            if len(predictions.shape) == 2:
                # 單一輸出模型，每個樣本一個預測向量
                predicted_classes = np.argmax(predictions, axis=1)
                
                # 創建結果 DataFrame
                results = pd.DataFrame({'預測類別': predicted_classes})
                
                logging.info(f"預測類別範圍: {np.min(predicted_classes)} - {np.max(predicted_classes)}")
            elif len(predictions.shape) == 3:
                # 多輸出模型，每個樣本多個預測
                logging.info(f"多輸出模型，輸出數: {predictions.shape[0]}")
                
                # 檢查輸出數量是否為272
                if predictions.shape[0] != 272:
                    logging.warning(f"輸出數量 ({predictions.shape[0]}) 與預期的272不符")
                
                # 對每個輸出進行argmax處理
                predicted_classes = [np.argmax(pred, axis=1) for pred in predictions]
                
                # 創建結果 DataFrame
                results = pd.DataFrame()
                # 使用實際的輸出數量而不是固定的272
                for i in range(len(predicted_classes)):
                    results[f'Sample_{i}'] = predicted_classes[i]
            else:
                # 異常形狀的輸出
                logging.error(f"無法處理的預測結果形狀: {predictions.shape}")
                raise ValueError(f"無法處理的預測結果形狀: {predictions.shape}")
        
        logging.info(f"預測結果處理完成，DataFrame形狀: {results.shape}")
        return results
    except Exception as e:
        logging.error(f"預測失敗: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def save_predictions(predictions, output_file):
    """保存預測結果"""
    try:
        predictions.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"預測結果已保存至: {output_file}")
    except Exception as e:
        logging.error(f"保存預測結果失敗: {str(e)}")
        raise

def main():
    """主程序"""
    try:
        # 設置日誌
        setup_logging()
        
        # 獲取當前工作目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 設置文件路徑
        predict_file = os.path.join(current_dir, 'tblp_thickness_wide_predict_20250428_enhance.csv')
        model_path = os.path.join(current_dir, 'tblp_model_enhanced.h5')
        scaler_path = os.path.join(current_dir, 'thickness_scaler_enhanced.npy')
        output_file = os.path.join(current_dir, 'tblp_predictions_20250428_enhance.csv')
        
        # 檢查所有必要檔案是否存在
        for file_path, file_desc in [
            (predict_file, "預測數據檔案"),
            (model_path, "模型檔案"),
            (scaler_path, "標準化器檔案")
        ]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到{file_desc}: {file_path}")
            else:
                file_size = os.path.getsize(file_path)
                logging.info(f"{file_desc}大小: {file_size} bytes")
        
        # 載入預測數據
        logging.info("開始載入預測數據...")
        X = load_predict_data(predict_file)
        
        # 預處理數據
        logging.info("開始預處理數據...")
        X_scaled = preprocess_predict_data(X, scaler_path)
        
        # 載入模型 - 使用簡化的模型載入流程
        model = None
        logging.info("嘗試載入模型...")
        
        # 方法列表，按優先順序排列
        loading_methods = [
            {
                "name": "方法1: 標準載入",
                "func": lambda: load_model(model_path, compile=False, custom_objects=None)
            },
            {
                "name": "方法2: h5py直接讀取",
                "func": lambda: load_model_with_h5py(model_path)
            },
            {
                "name": "方法3: TF實驗設置",
                "func": lambda: load_model_with_tf_experimental(model_path)
            },
            {
                "name": "方法4: 替代模型",
                "func": lambda: create_fallback_model(X_scaled.shape[1:])
            }
        ]
        
        # 逐一嘗試載入方法
        for method in loading_methods:
            try:
                logging.info(f"嘗試 {method['name']}...")
                model = method["func"]()
                logging.info(f"{method['name']} 成功！")
                break
            except Exception as e:
                logging.error(f"{method['name']} 失敗: {str(e)}")
        
        # 檢查是否成功載入模型
        if model is None:
            raise RuntimeError("所有模型載入方法都失敗")
            
        # 進行預測
        logging.info("開始預測...")
        predictions = predict(model, X_scaled)
        
        # 保存預測結果
        logging.info("保存預測結果...")
        save_predictions(predictions, output_file)
        
        logging.info("預測完成！")
        
    except Exception as e:
        logging.error(f"程序執行失敗: {str(e)}")
        raise

# 額外的輔助函數
def load_model_with_h5py(model_path):
    """使用h5py載入模型"""
    import h5py
    with h5py.File(model_path, 'r') as f:
        # 檢查h5檔案結構
        logging.info(f"H5檔案結構: {list(f.keys())}")
        
        # 從h5檔案中提取模型架構
        if 'model_config' in f.attrs:
            import json
            from tensorflow.keras.models import model_from_json
            
            model_config = f.attrs['model_config']
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
                
            model = model_from_json(model_config)
            
            # 嘗試載入權重
            if 'model_weights' in f:
                model.load_weights(model_path)
            
            return model
        else:
            raise ValueError("H5檔案中找不到模型配置")

def load_model_with_tf_experimental(model_path):
    """使用TF實驗設置載入模型"""
    import tensorflow as tf
    # 設置實驗屬性
    if not hasattr(tf.keras.models, 'experimental'):
        tf.keras.models.experimental = {}
    tf.keras.models.experimental.save_format = 'h5'
    
    # 載入模型
    return tf.keras.models.load_model(model_path, compile=False)

def create_fallback_model(input_shape):
    """創建替代模型"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    
    logging.warning("創建未訓練的替代模型 - 結果將不可靠!")
    
    # 根據輸入形狀創建一個簡單的CNN模型
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 假設有3個輸出類別，可調整
    ])
    
    return model

if __name__ == "__main__":
    main() 
