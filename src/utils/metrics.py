import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """计算各种评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 方向准确率
    direction_acc = np.mean((y_true * y_pred) > 0)
    
    return {
        'mse': mse,
        'mae': mae,
        'direction_acc': direction_acc
    }
