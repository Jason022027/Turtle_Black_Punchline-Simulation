import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sample_data_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def generate_sample_data(num_samples=1000):
    """生成示例數據"""
    try:
        # 生成TBLP狀態（0:關, 1:半開, 2:開）
        states = np.random.randint(0, 3, num_samples)
        
        # 生成膜厚數據
        thickness_data = []
        for state in states:
            if state == 0:  # 關
                base_thickness = np.random.normal(85, 2)  # 較厚
            elif state == 1:  # 半開
                base_thickness = np.random.normal(80, 2)  # 中等
            else:  # 開
                base_thickness = np.random.normal(75, 2)  # 較薄
            
            # 生成49個點位的膜厚值
            thickness = base_thickness + np.random.normal(0, 1, 49)
            thickness_data.append(thickness)
        
        # 創建DataFrame
        thickness_df = pd.DataFrame(thickness_data, columns=[f'Sample_{i}' for i in range(49)])
        states_df = pd.DataFrame({'TBLP_State': states})
        
        # 添加編號
        thickness_df['number'] = range(1, num_samples + 1)
        states_df['number'] = range(1, num_samples + 1)
        
        # 保存數據
        thickness_file = f'tblp_thickness_wide_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        states_file = f'tblp_states_wide_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        thickness_df.to_csv(thickness_file, index=False)
        states_df.to_csv(states_file, index=False)
        
        logging.info(f"已生成 {num_samples} 個樣本")
        logging.info(f"膜厚數據保存至: {thickness_file}")
        logging.info(f"狀態數據保存至: {states_file}")
        
        return thickness_file, states_file
    except Exception as e:
        logging.error(f"生成示例數據時發生錯誤: {str(e)}")
        raise

def generate_new_data(num_samples=10):
    """生成新的預測數據"""
    try:
        # 生成膜厚數據
        thickness_data = []
        for _ in range(num_samples):
            # 隨機生成基線膜厚
            base_thickness = np.random.normal(80, 5)  # 80±5 nm
            
            # 生成49個點位的膜厚值
            thickness = base_thickness + np.random.normal(0, 2, 49)
            thickness_data.append(thickness)
        
        # 創建DataFrame
        df = pd.DataFrame(thickness_data, columns=[f'Sample_{i}' for i in range(49)])
        df['number'] = range(1, num_samples + 1)
        
        # 保存數據
        output_file = f'new_thickness_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        
        logging.info(f"已生成 {num_samples} 個新樣本")
        logging.info(f"數據保存至: {output_file}")
        
        return output_file
    except Exception as e:
        logging.error(f"生成新數據時發生錯誤: {str(e)}")
        raise

def main():
    """主函數"""
    try:
        # 生成訓練數據
        thickness_file, states_file = generate_sample_data(1000)
        
        # 生成新的預測數據
        new_data_file = generate_new_data(10)
        
        logging.info("數據生成完成")
        
    except Exception as e:
        logging.error(f"程式執行時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 