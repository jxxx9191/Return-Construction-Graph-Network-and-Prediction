import torch
from torch.optim import Adam
import yaml
import os
import numpy as np
from data.data_processor import StockRelationBuilder
from data.dataset import StockDataset
from models.gnn import StockGNN
import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像时负号'-'显示为方块的问题

def visualize_stock_network(relations_df, features_df, save_path=None):
    """可视化股票关系网络"""
    print("\n构建并可视化股票关系网络...")
    
    # 创建无向图
    G = nx.Graph()
    
    # 添加节点
    unique_stocks = pd.concat([
        relations_df['source_stock'],
        relations_df['target_stock']
    ]).unique()
    
    # 获取每个股票的平均收益率作为节点大小
    stock_returns = features_df.groupby('stock_code')['return_rate'].mean()
    
    # 添加节点，设置节点大小基于收益率
    for stock in unique_stocks:
        size = abs(stock_returns.get(stock, 0)) * 1000  # 放大收益率作为节点大小
        G.add_node(stock, size=max(100, size))  # 设置最小大小为100
    
    # 添加边，权重基于相关系数
    for _, row in relations_df.iterrows():
        G.add_edge(
            row['source_stock'],
            row['target_stock'],
            weight=row['weight']
        )
    
    # 设置绘图参数
    plt.figure(figsize=(15, 10))
    
    # 使用spring_layout布局
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 获取节点大小和边权重
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # 绘制网络
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.7)
    
    nx.draw_networkx_edges(G, pos,
                          width=[w * 2 for w in edge_weights],
                          alpha=0.5,
                          edge_color='gray')
    
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("股票关系网络\n(节点大小表示收益率，边宽度表示相关性强度)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"网络图已保存至: {save_path}")
    
    plt.show()
    
    # 打印网络统计信息
    print("\n网络统计信息:")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    return G

def evaluate_predictions(model, loader, device):
    """评估模型预测结果"""
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, -1)
            
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals

def visualize_predictions(predictions, actuals, save_dir):
    """可视化预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 预测值vs实际值散点图
    plt.subplot(2, 2, 1)
    plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
    plt.xlabel('实际收益率')
    plt.ylabel('预测收益率')
    plt.title('预测值vs实际值对比')
    plt.legend()
    
    # 2. 预测误差分布直方图
    plt.subplot(2, 2, 2)
    errors = predictions - actuals
    plt.hist(errors.flatten(), bins=50, alpha=0.75)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    
    # 3. 时间序列预测
    plt.subplot(2, 1, 2)
    time_steps = range(len(predictions))
    plt.plot(time_steps, actuals[:, 0], label='实际值', alpha=0.7)
    plt.plot(time_steps, predictions[:, 0], label='预测值', alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('收益率')
    plt.title('收益率预测时间序列')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n预测结果可视化已保存至: {save_path}")
    plt.show()

def train_model():
    try:
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        print(f"加载配置文件: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功")

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 加载数据
        processed_file = r'E:\me\data\processed\processed_stocks.csv'
        relations_file = r'E:\me\data\processed\stock_relations.csv'
        
        print("\n加载数据...")
        features_df = pd.read_csv(processed_file)
        relations_df = pd.read_csv(relations_file)
        
        # 可视化股票关系网络
        network_plot_path = os.path.join(os.path.dirname(processed_file), 'stock_network.png')
        G = visualize_stock_network(relations_df, features_df, network_plot_path)
        
        # 使用部分数据进行测试
        features_df = features_df.head(1000)  # 先用小数据集测试
        
        print(f"特征数据形状: {features_df.shape}")
        print(f"关系数据形状: {relations_df.shape}")

        # 创建数据集
        print("\n创建数据集...")
        window_size = config['data']['window_size']
        prediction_days = config['data']['prediction_days']
        
        # 划分训练集和测试集
        train_size = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:train_size]
        test_df = features_df.iloc[train_size:]
        
        train_dataset = StockDataset(
            features_df=train_df,
            relations_df=relations_df,
            window_size=window_size,
            prediction_days=prediction_days
        )
        
        test_dataset = StockDataset(
            features_df=test_df,
            relations_df=relations_df,
            window_size=window_size,
            prediction_days=prediction_days
        )

        # 创建数据加载器
        batch_size = 8  # 使用更小的batch_size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"批次大小: {batch_size}")

        # 获取实际特征维度
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.x.shape[-1]
        print(f"\n实际特征维度: {input_dim}")

        # 创建模型
        print("\n创建模型...")
        model = StockGNN(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            output_dim=prediction_days,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        
        print("模型结构:")
        print(model)

        # 训练循环
        print("\n开始训练...")
        optimizer = Adam(model.parameters(), lr=config['training']['lr'])
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        losses = []

        for epoch in range(config['training']['epochs']):
            model.train()
            total_loss = 0
            batch_count = 0
            epoch_losses = []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    y = batch.y.view(batch.num_graphs, -1)
                    
                    loss = torch.nn.MSELoss()(out, y)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n批次 {batch_idx} 损失值无效: {loss.item()}")
                        continue
                    
                    loss.backward()
                    optimizer.step()
                    
                    loss_value = loss.item()
                    total_loss += loss_value
                    batch_count += 1
                    epoch_losses.append(loss_value)
                    
                    pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                    
                except Exception as e:
                    print(f"\n处理批次 {batch_idx} 时出错:")
                    print(f"错误信息: {str(e)}")
                    continue

            avg_loss = np.mean([l for l in epoch_losses if not (np.isnan(l) or np.isinf(l))])
            losses.append(avg_loss)

            print(f'\nEpoch {epoch}/{config["training"]["epochs"]}, '
                  f'Average Loss: {avg_loss:.4f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                model_save_path = os.path.join(os.path.dirname(processed_file), 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config,
                }, model_save_path)
                print(f"保存最佳模型，损失: {best_loss:.4f}")
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"\n{patience} 轮没有改善，停止训练")
                break

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('训练损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.yscale('log')
        plt.grid(True)
        loss_plot_path = os.path.join(os.path.dirname(processed_file), 'training_loss.png')
        plt.savefig(loss_plot_path)
        plt.show()

        # 评估模型预测效果
        print("\n评估模型预测效果...")
        predictions, actuals = evaluate_predictions(model, test_loader, device)
        
        # 计算评估指标
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        print("\n预测效果评估:")
        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"均方根误差 (RMSE): {rmse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        
        # 可视化预测结果
        save_dir = os.path.dirname(processed_file)
        visualize_predictions(predictions, actuals, save_dir)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            '实际收益率': actuals[:, 0],
            '预测收益率': predictions[:, 0],
            '预测误差': np.abs(predictions[:, 0] - actuals[:, 0])
        })
        
        results_path = os.path.join(save_dir, 'prediction_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存至: {results_path}")

        print("\n训练和评估完成!")
        print(f"模型保存在: {model_save_path}")
        print(f"预测可视化保存在: {os.path.join(save_dir, 'predictions.png')}")

    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == '__main__':
    train_model()