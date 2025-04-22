import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tblp_cnn_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_model_and_scaler():
    """載入模型和scaler"""
    try:
        model = tf.keras.models.load_model('tblp_cnn_model.h5')
        scale = np.load('thickness_scaler.npy')
        mean = np.load('thickness_mean.npy')
        logging.info("成功載入模型和scaler")
        return model, scale, mean
    except Exception as e:
        logging.error(f"載入模型時發生錯誤: {str(e)}")
        raise

def preprocess_input(thickness_values, scale, mean):
    """預處理輸入數據"""
    try:
        # 確保輸入是49個點位
        if len(thickness_values) != 49:
            raise ValueError("輸入必須包含49個點位的膜厚值")
        
        # 標準化
        thickness_scaled = (thickness_values - mean) / scale
        
        # 重塑為CNN輸入格式
        X = thickness_scaled.reshape(1, 49, 1)
        
        return X
    except Exception as e:
        logging.error(f"預處理輸入數據時發生錯誤: {str(e)}")
        raise

def predict_tblp_state(thickness_values):
    """預測TBLP狀態"""
    try:
        # 載入模型和scaler
        model, scale, mean = load_model_and_scaler()
        
        # 預處理輸入數據
        X = preprocess_input(thickness_values, scale, mean)
        
        # 進行預測
        predictions = model.predict(X)
        predicted_state = np.argmax(predictions[0])
        
        # 轉換為狀態描述
        state_map = {0: '關', 1: '半開', 2: '開'}
        state_description = state_map[predicted_state]
        
        logging.info(f"預測結果: TBLP狀態 = {state_description}")
        return predicted_state, state_description, predictions[0]
    except Exception as e:
        logging.error(f"預測時發生錯誤: {str(e)}")
        raise

def predict_from_csv(input_file):
    """從CSV文件讀取數據並進行預測"""
    try:
        # 讀取CSV文件
        df = pd.read_csv(input_file)
        
        # 提取49個點位的膜厚值
        thickness_columns = [f'Sample_{i}' for i in range(49)]
        thickness_values = df[thickness_columns].values
        
        # 載入模型和scaler
        model, scale, mean = load_model_and_scaler()
        
        # 預處理所有數據
        X = (thickness_values - mean) / scale
        X = X.reshape(-1, 49, 1)
        
        # 進行預測
        predictions = model.predict(X)
        predicted_states = np.argmax(predictions, axis=1)
        
        # 添加預測結果到DataFrame
        df['Predicted_State'] = predicted_states
        df['State_Description'] = df['Predicted_State'].map({0: '關', 1: '半開', 2: '開'})
        
        # 保存結果
        output_file = f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        
        logging.info(f"預測完成，結果已保存至: {output_file}")
        return df
    except Exception as e:
        logging.error(f"從CSV文件預測時發生錯誤: {str(e)}")
        raise

def main():
    """主函數"""
    try:
        # 示例：從CSV文件進行預測
        results = predict_from_csv('new_thickness_data.csv')
        print("\n預測結果摘要:")
        print(results[['Predicted_State', 'State_Description']].head())
        
    except Exception as e:
        logging.error(f"程式執行時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 