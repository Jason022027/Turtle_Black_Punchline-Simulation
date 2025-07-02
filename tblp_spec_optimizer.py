import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """設置日誌配置"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'tblp_optimizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_model_and_scalers(model_path):
    """載入訓練好的模型和標準化器"""
    try:
        # 載入模型
        model = tf.keras.models.load_model(model_path)
        logging.info(f"成功載入模型: {model_path}")
        
        # 載入標準化器
        scaler_X = np.load('delta_tblp_scaler.npy', allow_pickle=True).item()
        scaler_y = np.load('delta_thickness_scaler.npy', allow_pickle=True).item()
        logging.info("成功載入標準化器")
        
        return model, scaler_X, scaler_y
    
    except Exception as e:
        logging.error(f"載入模型或標準化器失敗: {str(e)}")
        raise

def calculate_thickness_metrics(thickness_array):
    """計算膜厚統計指標"""
    avg = np.mean(thickness_array)
    std = np.std(thickness_array)
    return avg, std

def predict_thickness_change(model, delta_tblp, scaler_X, scaler_y):
    """預測膜厚變化"""
    try:
        # 標準化輸入
        delta_tblp_scaled = scaler_X.transform(delta_tblp.reshape(1, -1))
        
        # 預測
        delta_thickness_scaled = model.predict(delta_tblp_scaled, verbose=0)
        
        # 反標準化
        delta_thickness = scaler_y.inverse_transform(delta_thickness_scaled)
        
        return delta_thickness.flatten()
    
    except Exception as e:
        logging.error(f"預測失敗: {str(e)}")
        raise

def objective_function(delta_tblp, current_thickness, target_avg, target_std_max, model, scaler_X, scaler_y, weights=None):
    """目標函數：最小化與目標 SPEC 的差距"""
    if weights is None:
        weights = {'avg': 1.0, 'std': 1.0, 'change': 0.1}
    
    try:
        # 預測膜厚變化
        predicted_delta = predict_thickness_change(model, delta_tblp, scaler_X, scaler_y)
        
        # 計算預測的新膜厚
        new_thickness = current_thickness + predicted_delta
        
        # 計算新的統計指標
        new_avg, new_std = calculate_thickness_metrics(new_thickness)
        
        # 計算與目標的差距
        avg_error = abs(new_avg - target_avg)
        std_penalty = max(0, new_std - target_std_max)
        
        # 懲罰過大的 TBLP 變化
        change_penalty = np.sum(np.abs(delta_tblp))
        
        # 組合目標函數
        total_cost = (weights['avg'] * avg_error + 
                     weights['std'] * std_penalty + 
                     weights['change'] * change_penalty)
        
        return total_cost
    
    except Exception as e:
        return 1e6  # 返回一個很大的值表示失敗

def find_optimal_tblp_adjustment(current_thickness, target_avg=85.0, target_std_max=3.0, 
                                model=None, scaler_X=None, scaler_y=None, 
                                max_change=2, weights=None):
    """找出最佳的 TBLP 調整方案"""
    if weights is None:
        weights = {'avg': 1.0, 'std': 1.0, 'change': 0.1}
    
    logging.info(f"開始搜尋最佳 TBLP 調整...")
    logging.info(f"目標平均值: {target_avg}")
    logging.info(f"目標最大標準差: {target_std_max}")
    logging.info(f"允許的最大 TBLP 變化: ±{max_change}")
    
    # 當前膜厚統計
    current_avg, current_std = calculate_thickness_metrics(current_thickness)
    logging.info(f"當前膜厚 - 平均: {current_avg:.3f}, 標準差: {current_std:.3f}")
    
    # 設置初始猜測（零變化）
    x0 = np.zeros(272)
    
    # 設置邊界條件（TBLP 變化限制在 -max_change 到 +max_change）
    bounds = [(-max_change, max_change) for _ in range(272)]
    
    # 定義目標函數的包裝器
    def objective_wrapper(delta_tblp):
        return objective_function(delta_tblp, current_thickness, target_avg, target_std_max, 
                                model, scaler_X, scaler_y, weights)
    
    # 多次隨機初始化優化
    best_result = None
    best_cost = np.inf
    
    for i in range(5):  # 嘗試5次不同的初始化
        if i > 0:
            # 隨機初始化
            x0 = np.random.uniform(-0.5, 0.5, 272)
        
        try:
            result = minimize(
                objective_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False}
            )
            
            if result.success and result.fun < best_cost:
                best_result = result
                best_cost = result.fun
                
        except Exception as e:
            logging.warning(f"優化嘗試 {i+1} 失敗: {str(e)}")
            continue
    
    if best_result is None:
        logging.error("所有優化嘗試都失敗")
        return None, None, None
    
    # 獲取最佳解
    optimal_delta_tblp = best_result.x
    
    # 預測結果
    predicted_delta = predict_thickness_change(model, optimal_delta_tblp, scaler_X, scaler_y)
    predicted_thickness = current_thickness + predicted_delta
    predicted_avg, predicted_std = calculate_thickness_metrics(predicted_thickness)
    
    # 記錄結果
    logging.info(f"優化完成！")
    logging.info(f"預測新膜厚 - 平均: {predicted_avg:.3f}, 標準差: {predicted_std:.3f}")
    logging.info(f"TBLP 變化範圍: [{optimal_delta_tblp.min():.3f}, {optimal_delta_tblp.max():.3f}]")
    logging.info(f"非零 TBLP 變化數量: {np.sum(np.abs(optimal_delta_tblp) > 0.01)}")
    
    return optimal_delta_tblp, predicted_thickness, {
        'current_avg': current_avg,
        'current_std': current_std,
        'predicted_avg': predicted_avg,
        'predicted_std': predicted_std,
        'meets_spec': predicted_avg >= target_avg - 0.5 and predicted_avg <= target_avg + 0.5 and predicted_std <= target_std_max
    }

def visualize_optimization_results(current_thickness, predicted_thickness, delta_tblp, results_info):
    """視覺化優化結果"""
    try:
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 膜厚分布比較
        axes[0, 0].plot(current_thickness, 'o-', label=f'Current (avg={results_info["current_avg"]:.2f}, std={results_info["current_std"]:.2f})', alpha=0.7)
        axes[0, 0].plot(predicted_thickness, 's-', label=f'Predicted (avg={results_info["predicted_avg"]:.2f}, std={results_info["predicted_std"]:.2f})', alpha=0.7)
        axes[0, 0].axhline(y=85.0, color='r', linestyle='--', alpha=0.5, label='Target Avg')
        axes[0, 0].set_xlabel('膜厚點位置')
        axes[0, 0].set_ylabel('膜厚值')
        axes[0, 0].set_title('膜厚分布比較')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 膜厚變化
        delta_thickness = predicted_thickness - current_thickness
        axes[0, 1].bar(range(len(delta_thickness)), delta_thickness, alpha=0.7)
        axes[0, 1].set_xlabel('膜厚點位置')
        axes[0, 1].set_ylabel('膜厚變化')
        axes[0, 1].set_title('各點膜厚變化')
        axes[0, 1].grid(True)
        
        # 3. TBLP 調整分布
        significant_changes = np.abs(delta_tblp) > 0.01
        axes[1, 0].bar(range(len(delta_tblp)), delta_tblp, alpha=0.7)
        axes[1, 0].set_xlabel('TBLP 位置')
        axes[1, 0].set_ylabel('TBLP 變化')
        axes[1, 0].set_title(f'TBLP 調整 (共 {np.sum(significant_changes)} 個顯著變化)')
        axes[1, 0].grid(True)
        
        # 4. 統計指標比較
        metrics = ['平均值', '標準差']
        current_values = [results_info['current_avg'], results_info['current_std']]
        predicted_values = [results_info['predicted_avg'], results_info['predicted_std']]
        targets = [85.0, 3.0]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        axes[1, 1].bar(x - width, current_values, width, label='Current', alpha=0.7)
        axes[1, 1].bar(x, predicted_values, width, label='Predicted', alpha=0.7)
        axes[1, 1].bar(x + width, targets, width, label='Target', alpha=0.7)
        
        axes[1, 1].set_xlabel('指標')
        axes[1, 1].set_ylabel('值')
        axes[1, 1].set_title('統計指標比較')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 添加達標狀態
        status = "✓ 達標" if results_info['meets_spec'] else "✗ 未達標"
        fig.suptitle(f'TBLP 優化結果 - {status}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("優化結果圖已保存")
        
    except Exception as e:
        logging.error(f"視覺化失敗: {str(e)}")

def main():
    """主程序"""
    try:
        setup_logging()
        logging.info("=" * 60)
        logging.info("TBLP SPEC Optimizer Started")
        logging.info("目標：根據 SPEC 要求找出最佳 TBLP 調整")
        logging.info("=" * 60)
        
        # 載入模型（你需要指定實際的模型路徑）
        model_files = [f for f in os.listdir('.') if f.startswith('tblp_forward_model_') and f.endswith('.h5')]
        if not model_files:
            logging.error("找不到訓練好的模型檔案")
            return
        
        latest_model = sorted(model_files)[-1]  # 使用最新的模型
        logging.info(f"使用模型: {latest_model}")
        
        model, scaler_X, scaler_y = load_model_and_scalers(latest_model)
        
        # 載入測試數據（或者你可以手動輸入當前膜厚）
        thickness_file = 'tblp_thickness_wide_test_20250428.csv'
        if os.path.exists(thickness_file):
            thickness_df = pd.read_csv(thickness_file)
            thickness_columns = [f'Sample_{i}' for i in range(49)]
            thickness_data = thickness_df[thickness_columns].values
            
            # 使用第一筆數據作為示例
            current_thickness = thickness_data[0]
            logging.info("使用數據檔案中的第一筆膜厚數據作為示例")
        else:
            # 生成示例數據
            np.random.seed(42)
            current_thickness = np.random.normal(83, 4, 49)  # 模擬一個標準差過大的情況
            logging.info("使用模擬的膜厚數據作為示例")
        
        # 設置目標 SPEC
        target_avg = 85.0
        target_std_max = 3.0
        
        # 設置權重（可調整）
        weights = {
            'avg': 2.0,      # 平均值的重要性
            'std': 3.0,      # 標準差的重要性
            'change': 0.1    # 變化最小化的重要性
        }
        
        # 尋找最佳調整
        optimal_delta, predicted_thickness, results = find_optimal_tblp_adjustment(
            current_thickness=current_thickness,
            target_avg=target_avg,
            target_std_max=target_std_max,
            model=model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            max_change=2,  # 允許的最大 TBLP 變化
            weights=weights
        )
        
        if optimal_delta is not None:
            # 視覺化結果
            visualize_optimization_results(current_thickness, predicted_thickness, optimal_delta, results)
            
            # 保存建議調整
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 找出需要調整的 TBLP 位置
            significant_changes = np.abs(optimal_delta) > 0.01
            adjustment_suggestions = []
            
            for i, change in enumerate(optimal_delta):
                if abs(change) > 0.01:
                    direction = "增加" if change > 0 else "減少"
                    adjustment_suggestions.append({
                        'TBLP_Position': i,
                        'Change_Amount': change,
                        'Direction': direction,
                        'Magnitude': abs(change)
                    })
            
            # 按變化量排序
            adjustment_suggestions.sort(key=lambda x: x['Magnitude'], reverse=True)
            
            # 保存建議
            suggestions_df = pd.DataFrame(adjustment_suggestions)
            suggestions_file = f'tblp_adjustment_suggestions_{timestamp}.csv'
            suggestions_df.to_csv(suggestions_file, index=False)
            
            logging.info(f"調整建議已保存到: {suggestions_file}")
            logging.info(f"總共需要調整 {len(adjustment_suggestions)} 個 TBLP 位置")
            
            # 顯示前10個最重要的調整
            logging.info("前10個最重要的調整:")
            for i, suggestion in enumerate(adjustment_suggestions[:10]):
                logging.info(f"  {i+1}. TBLP_{suggestion['TBLP_Position']}: {suggestion['Direction']} {suggestion['Magnitude']:.3f}")
        
        logging.info("=" * 60)
        logging.info("優化完成！")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"程序執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 