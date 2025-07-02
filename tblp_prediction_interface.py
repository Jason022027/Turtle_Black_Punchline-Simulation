import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import logging

def setup_simple_logging():
    """簡單的日誌設置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def load_model_and_scalers():
    """載入最新的模型和標準化器"""
    try:
        # 尋找最新的模型
        model_files = [f for f in os.listdir('.') if f.startswith('tblp_forward_model_') and f.endswith('.h5')]
        if not model_files:
            print("錯誤：找不到訓練好的模型檔案")
            return None, None, None
        
        latest_model = sorted(model_files)[-1]
        print(f"載入模型: {latest_model}")
        
        model = tf.keras.models.load_model(latest_model)
        scaler_X = np.load('delta_tblp_scaler.npy', allow_pickle=True).item()
        scaler_y = np.load('delta_thickness_scaler.npy', allow_pickle=True).item()
        
        print("模型和標準化器載入成功！")
        return model, scaler_X, scaler_y
    
    except Exception as e:
        print(f"載入失敗: {str(e)}")
        return None, None, None

def predict_thickness_from_tblp_change(delta_tblp, model, scaler_X, scaler_y):
    """根據 TBLP 變化預測膜厚變化"""
    try:
        # 標準化輸入
        delta_tblp_scaled = scaler_X.transform(delta_tblp.reshape(1, -1))
        
        # 預測
        delta_thickness_scaled = model.predict(delta_tblp_scaled, verbose=0)
        
        # 反標準化
        delta_thickness = scaler_y.inverse_transform(delta_thickness_scaled)
        
        return delta_thickness.flatten()
    
    except Exception as e:
        print(f"預測失敗: {str(e)}")
        return None

def simple_tblp_suggestion(current_thickness, target_avg=85.0, target_std_max=3.0):
    """簡單的 TBLP 調整建議（基於經驗法則）"""
    current_avg = np.mean(current_thickness)
    current_std = np.std(current_thickness)
    
    print(f"\n當前膜厚統計:")
    print(f"  平均值: {current_avg:.3f}")
    print(f"  標準差: {current_std:.3f}")
    print(f"\n目標規格:")
    print(f"  平均值: {target_avg}")
    print(f"  標準差: < {target_std_max}")
    
    suggestions = []
    
    # 平均值調整建議
    avg_diff = target_avg - current_avg
    if abs(avg_diff) > 0.5:
        if avg_diff > 0:
            suggestions.append(f"建議整體增加膜厚約 {avg_diff:.2f} 單位")
        else:
            suggestions.append(f"建議整體減少膜厚約 {abs(avg_diff):.2f} 單位")
    
    # 標準差調整建議
    if current_std > target_std_max:
        suggestions.append(f"當前標準差過高({current_std:.3f})，需要改善均勻性")
        
        # 找出離群點
        deviation_from_mean = np.abs(current_thickness - current_avg)
        outliers = np.where(deviation_from_mean > current_std)[0]
        
        if len(outliers) > 0:
            suggestions.append(f"特別注意位置: {outliers.tolist()}")
    
    return suggestions

def main():
    """主程序"""
    setup_simple_logging()
    
    print("=" * 60)
    print("TBLP 膜厚預測介面")
    print("=" * 60)
    
    # 載入模型
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    if model is None:
        print("無法載入模型，僅提供簡單建議")
        use_model = False
    else:
        use_model = True
    
    while True:
        print("\n選擇輸入方式:")
        print("1. 從 CSV 檔案載入膜厚數據")
        print("2. 手動輸入膜厚數據（示例）")
        print("3. 退出")
        
        choice = input("請選擇 (1-3): ").strip()
        
        if choice == '3':
            break
        elif choice == '1':
            # 從檔案載入
            filename = input("請輸入 CSV 檔案名稱 (或按 Enter 使用預設): ").strip()
            if not filename:
                filename = 'tblp_thickness_wide_test_20250428.csv'
            
            try:
                df = pd.read_csv(filename)
                thickness_columns = [f'Sample_{i}' for i in range(49)]
                thickness_data = df[thickness_columns].values
                
                print(f"成功載入 {len(thickness_data)} 筆膜厚數據")
                
                # 選擇使用哪一筆數據
                idx = int(input(f"選擇要分析的數據索引 (0-{len(thickness_data)-1}): "))
                current_thickness = thickness_data[idx]
                
            except Exception as e:
                print(f"載入檔案失敗: {str(e)}")
                continue
                
        elif choice == '2':
            # 使用示例數據
            print("使用示例膜厚數據...")
            np.random.seed(42)
            current_thickness = np.random.normal(83, 4, 49)  # 模擬不達標的情況
            
        else:
            print("無效選擇，請重試")
            continue
        
        # 顯示當前膜厚統計
        print(f"\n分析膜厚數據:")
        print(f"數據範圍: [{current_thickness.min():.2f}, {current_thickness.max():.2f}]")
        
        # 獲取簡單建議
        suggestions = simple_tblp_suggestion(current_thickness)
        
        print(f"\n基本建議:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
        # 如果有模型，提供更詳細的預測
        if use_model:
            print(f"\n是否要使用 AI 模型進行詳細預測？ (y/n): ", end="")
            if input().lower() == 'y':
                # 這裡可以整合優化演算法
                print("模型預測功能開發中...")
                print("請使用 tblp_spec_optimizer.py 進行完整的優化分析")
        
        print(f"\n" + "="*40)

if __name__ == "__main__":
    main() 