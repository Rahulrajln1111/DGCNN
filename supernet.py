import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import add_self_loops, degree

# --- Operation Space Definitions ---
class SampleOp(nn.Module):
    def __init__(self, op_type, k=20):
        super().__init__()
        self.op_type = op_type
        self.k = k

    def forward(self, x, batch=None):
        if self.op_type == 'knn':
            return knn_graph(x, k=self.k, batch=batch)
        elif self.op_type == 'dense':
            # Simplified for simulation; implement other samplers based on needs
            return knn_graph(x, k=self.k*2, batch=batch) 
        return None

class AggregateOp(MessagePassing):
    def __init__(self, aggr_type):
        # Allowable aggregations: 'max', 'mean', 'add'
        super().__init__(aggr=aggr_type)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class CombineOp(nn.Module):
    def __init__(self, in_channels, out_channels, combine_type):
        super().__init__()
        self.combine_type = combine_type
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, aggr_out):
        if self.combine_type == 'concat':
            out = torch.cat([x, aggr_out], dim=1)
        else: # 'add'
            out = x + aggr_out
        return self.mlp(out)

# --- SuperNet Construction ---
class GNNSuperNet(nn.Module):
    def __init__(self, num_positions=12, in_channels=3, out_channels=40):
        super().__init__()
        self.num_positions = num_positions # Configured to 12 as per HGNAS paper
        self.hidden_dim = 64
        
        # Operation Spaces mapped to positions
        self.samplers = nn.ModuleList([SampleOp('knn') for _ in range(num_positions)])
        self.aggregators = nn.ModuleList([AggregateOp('max') for _ in range(num_positions)])
        self.combiners = nn.ModuleList([CombineOp(self.hidden_dim * 2, self.hidden_dim, 'concat') for _ in range(num_positions)])
        
        self.input_mlp = nn.Linear(in_channels, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, out_channels)

    def forward(self, data, architecture_encoding):
        # architecture_encoding dictates which operations are active per position
        x, batch = data.x, data.batch
        x = self.input_mlp(x)
        
        for pos in range(self.num_positions):
            # Parse operations chosen for this position by the EA
            samp_idx, aggr_idx, comb_idx = architecture_encoding[pos]
            
            # 1. Sample
            edge_index = self.samplers[pos](x, batch)
            # 2. Aggregate
            aggr_out = self.aggregators[pos](x, edge_index)
            # 3. Combine
            x = self.combiners[pos](x, aggr_out)
            
        return F.log_softmax(self.classifier(x), dim=-1)
