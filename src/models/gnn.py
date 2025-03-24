import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class StockGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(StockGNN, self).__init__()
        
        print(f"模型参数: input_dim={input_dim}, hidden_dim={hidden_dim}, "
              f"output_dim={output_dim}, num_layers={num_layers}")
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 添加批归一化
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # 修改输出层
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 打印输入维度（用于调试）
        if not hasattr(self, 'first_forward'):
            print(f"输入维度: x={x.shape}, edge_index={edge_index.shape}, batch={batch.shape}")
            self.first_forward = True
        
        # 图卷积层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # 最后一层卷积
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = torch.relu(x)
        
        # 使用全局池化得到图级别的表示
        x = global_mean_pool(x, batch)
        
        # 输出层
        out = self.linear(x)
        
        # 确保输出维度正确
        if not hasattr(self, 'first_output'):
            print(f"输出维度: {out.shape}")
            print(f"目标维度: {data.y.shape}")
            self.first_output = True
        
        return out
