import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import os

class StockRelationBuilder:
    def __init__(self, 
                 correlation_threshold: float = 0.5,
                 rolling_window: int = 60,  
                 min_periods: int = 30):    
        self.correlation_threshold = correlation_threshold
        self.rolling_window = rolling_window
        self.min_periods = min_periods
        self.scaler = StandardScaler()
        
    def load_and_process_data(self, file_path: str) -> pd.DataFrame:
        """加载并预处理股票数据"""
        print(f"正在从 {file_path} 加载数据...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'stock_code'])
        
        # 检查必要的列
        required_columns = ['date', 'stock_code', 'return_rate', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        print(f"数据加载完成，形状: {df.shape}")
        return df
    
    def check_stationarity(self, series: pd.Series) -> bool:
        """进行ADF平稳性检验"""
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p值小于0.05认为是平稳的
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征并进行平稳性检验"""
        print("\n创建特征...")
        
        # 打印原始数据统计信息，用于调试
        print("原始return_rate的统计信息：")
        print(df['return_rate'].describe())
        
        # 确保数据按照正确顺序排序
        df = df.sort_values(['stock_code', 'date']).copy()  # 添加.copy()避免警告
        
        # 计算技术指标
        for days in [5, 10, 20]:
            # 修改移动平均计算方式
            df[f'return_{days}d'] = df.groupby('stock_code')['return_rate'].transform(
                lambda x: x.rolling(window=days, min_periods=1).mean()
            )
            
            df[f'vol_{days}d'] = df.groupby('stock_code')['volume'].transform(
                lambda x: x.rolling(window=days, min_periods=1).mean()
            )
        
        # 打印计算结果，用于调试
        print("\n计算后的return_5d统计信息：")
        print(df['return_5d'].describe())
        
        # 对每只股票进行平稳性检验
        stationarity_results = {}
        for stock in df['stock_code'].unique():
            stock_data = df[df['stock_code'] == stock]
            if len(stock_data['return_rate'].dropna()) > 0:  # 添加数据检查
                is_stationary = self.check_stationarity(stock_data['return_rate'].dropna())
                stationarity_results[stock] = is_stationary
            
        print("\n平稳性检验结果:")
        print(f"平稳的股票数量: {sum(stationarity_results.values())}")
        print(f"非平稳的股票数量: {len(stationarity_results) - sum(stationarity_results.values())}")
        
        # 修改填充方式，使用新的方法
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df.groupby('stock_code')[col].transform(lambda x: x.ffill())  # 使用ffill()
            df[col] = df.groupby('stock_code')[col].transform(lambda x: x.bfill())  # 使用bfill()
        
        # 最后填充剩余的NA值
        df = df.fillna(0)
        
        return df
    
    def calculate_rolling_correlation(self, return_matrix: pd.DataFrame) -> pd.DataFrame:
        """计算滚动相关系数矩阵（优化版本）"""
        print("\n计算滚动相关系数...")
        print(f"数据形状: {return_matrix.shape}")
        
        try:
            # 1. 只使用最近的数据，并确保数据完整性
            recent_data = return_matrix.iloc[-self.rolling_window:]
            # 移除包含太多缺失值的列
            valid_columns = recent_data.columns[recent_data.notna().sum() >= self.min_periods]
            recent_data = recent_data[valid_columns]
            
            print(f"使用最近 {self.rolling_window} 天的数据计算相关系数")
            print(f"有效股票数量: {len(valid_columns)}")
            
            # 2. 直接计算完整的相关系数矩阵
            corr_matrix = recent_data.corr()
            
            # 3. 移除自相关
            np.fill_diagonal(corr_matrix.values, 0)
            
            print("相关系数计算完成！")
            return corr_matrix
            
        except Exception as e:
            print(f"计算相关系数时出错: {str(e)}")
            raise
    
    def build_relations(self, 
                       price_data: pd.DataFrame, 
                       plot_correlation: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构建股票关系"""
        print("\n构建股票关系网络...")
        
        # 数据标准化
        price_data['return_rate_scaled'] = self.scaler.fit_transform(
            price_data[['return_rate']]
        )
        
        # 构建相关性矩阵
        return_matrix = price_data.pivot(
            columns='stock_code',
            values='return_rate_scaled',
            index='date'
        )
        
        corr_matrix = self.calculate_rolling_correlation(return_matrix)
        
        if plot_correlation:
            self.plot_correlation_matrix(corr_matrix)
        
        # 构建边列表
        relations = []
        stock_codes = corr_matrix.index
        
        for i in range(len(stock_codes)):
            for j in range(i+1, len(stock_codes)):
                corr = corr_matrix.iloc[i,j]
                if abs(corr) >= self.correlation_threshold:
                    relations.append({
                        'source_stock': stock_codes[i],
                        'target_stock': stock_codes[j],
                        'weight': abs(corr)
                    })
        
        relations_df = pd.DataFrame(relations)
        self.print_statistics(relations_df, corr_matrix)
        
        return relations_df, corr_matrix
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame):
        """绘制相关系数矩阵热力图"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, 
                   vmax=1)
        plt.title('Stock Return Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self, relations_df: pd.DataFrame, corr_matrix: pd.DataFrame):
        """打印关系统计信息"""
        print("\n=== 关系统计信息 ===")
        print(f"总股票数量: {len(corr_matrix)}")
        print(f"总关系数量: {len(relations_df)}")
        print(f"平均每只股票的关系数: {len(relations_df) * 2 / len(corr_matrix):.2f}")
        print("\n关系权重统计:")
        print(relations_df['weight'].describe())
    
    def process_and_build_relations(self, input_file: str, output_dir: str):
        """完整的处理流程"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载和处理数据
            df = self.load_and_process_data(input_file)
            df = self.create_features(df)
            
            # 构建关系
            relations_df, corr_matrix = self.build_relations(df, plot_correlation=True)
            
            # 保存结果
            processed_file = os.path.join(output_dir, 'processed_stocks.csv')
            relations_file = os.path.join(output_dir, 'stock_relations.csv')
            corr_matrix_file = os.path.join(output_dir, 'correlation_matrix.csv')
            
            df.to_csv(processed_file, index=False)
            relations_df.to_csv(relations_file, index=False)
            corr_matrix.to_csv(corr_matrix_file)
            
            print("\n数据处理和关系构建完成！")
            print(f"处理后的数据保存至: {processed_file}")
            print(f"关系数据保存至: {relations_file}")
            print(f"相关矩阵保存至: {corr_matrix_file}")
            
        except Exception as e:
            print(f"错误: {str(e)}")

def main():
    # 设置路径
    input_file = r'E:\me\data\raw\stock_prices.csv'
    output_dir = r'E:\me\data\processed'
    
    # 创建处理器实例，使用更小的窗口大小
    builder = StockRelationBuilder(
        correlation_threshold=0.5,
        rolling_window=30,  # 减小窗口大小
        min_periods=15     # 减小最小周期
    )
    
    # 运行处理流程
    builder.process_and_build_relations(input_file, output_dir)

if __name__ == "__main__":
    main()
