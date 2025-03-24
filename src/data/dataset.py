import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class StockDataset(Dataset):
    def __init__(self, features_df, relations_df, window_size, prediction_days):
        super(StockDataset, self).__init__()
        
        # 保存参数
        self.window_size = window_size
        self.prediction_days = prediction_days
        
        # 预处理特征数据
        print("预处理特征数据...")
        self.features_df = features_df.copy()
        self.features = self._preprocess_features(features_df)
        
        # 预处理关系数据
        print("预处理关系数据...")
        self.relations_df = relations_df.copy()
        
        # 获取所有唯一的股票代码
        self.stock_codes = sorted(features_df['stock_code'].unique())
        self.stock_to_idx = {code: idx for idx, code in enumerate(self.stock_codes)}
        
        print(f"数据集初始化完成:")
        print(f"股票数量: {len(self.stock_codes)}")
        print(f"时间窗口大小: {window_size}")
        print(f"预测天数: {prediction_days}")
        
    def _preprocess_features(self, df):
        """预处理特征数据"""
        # 选择数值类型的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].copy()
        
        # 处理无限值和NaN
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 确保所有数据为float32类型
        df_numeric = df_numeric.astype(np.float32)
        
        print(f"特征维度: {df_numeric.shape}")
        print(f"特征列: {df_numeric.columns.tolist()}")
        
        return df_numeric
        
    def _get_window_graph(self, window_data):
        """为当前时间窗口构建图结构"""
        try:
            # 获取当前窗口中的股票代码
            current_stocks = sorted(window_data['stock_code'].unique())
            current_stock_to_idx = {code: idx for idx, code in enumerate(current_stocks)}
            
            # 创建完全连接的图（每个节点都与其他节点相连）
            num_stocks = len(current_stocks)
            source_nodes = []
            target_nodes = []
            edge_weights = []
            
            # 创建完全连接的边
            for i in range(num_stocks):
                for j in range(num_stocks):
                    if i != j:  # 不包括自环
                        source_nodes.append(i)
                        target_nodes.append(j)
                        edge_weights.append(1.0)  # 默认权重为1
            
            # 如果有边
            if source_nodes:
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty(0, dtype=torch.float32)
            
            return edge_index, edge_attr
            
        except Exception as e:
            print(f"构建图结构时出错: {str(e)}")
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    
    def len(self):
        """返回数据集长度"""
        return len(self.features_df) - self.window_size - self.prediction_days
        
    def get(self, idx):
        """获取单个数据样本"""
        try:
        # 获取窗口数据
            window_data = self.features_df.iloc[idx:idx+self.window_size]
            target_data = self.features_df.iloc[idx+self.window_size:idx+self.window_size+self.prediction_days]
        
        # 检查数据是否足够
            if len(window_data) < self.window_size or len(target_data) < self.prediction_days:
                print(f"索引 {idx} 的数据不足，跳过")
                return self.get(0)  # 返回第一个有效样本
        
        # 获取当前窗口的图结构
            edge_index, edge_attr = self._get_window_graph(window_data)
        
        # 获取特征数据（排除 'date' 和 'stock_code' 列）
            feature_cols = [col for col in self.features.columns if col not in ['date', 'stock_code']]
            feature_data = self.features[feature_cols].iloc[idx:idx+self.window_size]
        
        # 转换为张量
            x = torch.tensor(feature_data.values, dtype=torch.float32)
        
        # 获取所有预测天数的目标值，并重塑为 (prediction_days, 1) 形状
            target_values = target_data['return_rate'].values[:self.prediction_days]
            y = torch.tensor(target_values, dtype=torch.float32).reshape(-1, 1)  # 改为 (prediction_days, 1)
        
        # 检查数据的有效性
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"索引 {idx} 的数据包含 NaN 值，使用第一个有效样本替代")
                return self.get(0)
        
        # 检查维度
            if idx == 0:
                print(f"\n第一个样本的维度:")
                print(f"特征维度 (x): {x.shape}")  # 应该是 [window_size, n_features]
                print(f"目标维度 (y): {y.shape}")  # 现在应该是 [prediction_days, 1]
                print(f"边索引维度: {edge_index.shape}")
                print(f"边属性维度: {edge_attr.shape}")
                print(f"特征列: {feature_cols}")
                print(f"目标值: {y}")  # 打印实际的目标值
        
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )
        
        except Exception as e:
            print(f"处理索引 {idx} 时出错:")
            print(f"错误信息: {str(e)}")
            return self.get(0)

def main():
    """测试数据集的功能"""
    try:
        # 设置数据路径
        processed_file = r'E:\me\data\processed\processed_stocks.csv'
        relations_file = r'E:\me\data\processed\stock_relations.csv'
        
        # 加载数据
        print("加载数据...")
        features_df = pd.read_csv(processed_file)
        relations_df = pd.read_csv(relations_file)
        
        # 使用部分数据进行测试
        features_df = features_df.head(1000)  # 先用小数据集测试
        
        print("数据形状:")
        print(f"特征数据: {features_df.shape}")
        print(f"关系数据: {relations_df.shape}")
        
        # 创建数据集
        dataset = StockDataset(
            features_df=features_df,
            relations_df=relations_df,
            window_size=10,
            prediction_days=5
        )

        # 测试单个数据获取
        print("\n测试单个数据获取...")
        for i in tqdm(range(min(5, len(dataset)))):
            sample = dataset.get(i)
            if i == 0:
                print("\n第一个样本信息:")
                print(f"特征形状: {sample.x.shape}")
                print(f"边索引形状: {sample.edge_index.shape}")
                print(f"边属性形状: {sample.edge_attr.shape}")
                print(f"目标形状: {sample.y.shape}")
        
        # 测试批处理数据加载
        print("\n测试批处理数据加载...")
        
        # 创建数据加载器
        batch_size = 8  # 使用更小的batch_size
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,  # 丢弃不完整的最后一个批次
            num_workers=0    # 不使用多进程加载
        )
        
        # 测试批处理
        print(f"\n测试第一个批次...")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"批次数据信息:")
                print(f"批次特征形状: {batch.x.shape}")
                print(f"批次边索引形状: {batch.edge_index.shape}")
                print(f"批次边属性形状: {batch.edge_attr.shape}")
                print(f"批次目标形状: {batch.y.shape}")
                print(f"批次大小: {batch.num_graphs}")
            break  # 只测试第一个批次

    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()