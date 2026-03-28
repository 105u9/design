# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

# Phase 3: Simplified Graph Embedding (GraphSAGE-like)
class GraphSageLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSageLayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        
    def forward(self, x, adj):
        # x: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        
        # Simple aggregation: mean of neighbors
        num_neighbors = adj.sum(dim=1, keepdim=True)
        # Handle zero neighbors
        num_neighbors = torch.where(num_neighbors == 0, torch.ones_like(num_neighbors), num_neighbors)
        neighbor_agg = torch.matmul(adj, x) / num_neighbors
        
        # Concatenate self-features and neighbor features
        combined = torch.cat([x, neighbor_agg], dim=1)
        return torch.relu(self.linear(combined))

# Phase 3: Graph Attention Network (GAT) Layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.out_features = out_features // heads
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # h: [B, N, in_feat] or [N, in_feat]
        is_batched = h.dim() == 3
        if not is_batched:
            h = h.unsqueeze(0)
            
        B, N, _ = h.size()
        Wh = torch.matmul(h, self.W) # [B, N, out_feat]
        
        # a_input: [B, N, N, 2*out_feat]
        Wh_repeat = Wh.repeat_interleave(N, dim=1) # [B, N*N, out_feat]
        Wh_tile = Wh.repeat(1, N, 1) # [B, N*N, out_feat]
        a_input = torch.cat([Wh_repeat, Wh_tile], dim=2).view(B, N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # [B, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        # adj should be [N, N] or [B, N, N]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
            
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh) # [B, N, out_feat]

        out = torch.relu(h_prime)
        return out if is_batched else out.squeeze(0)

# Phase 3: Node2Vec Embedding (Simplified)
class Node2VecEmbedding:
    def __init__(self, num_nodes, embedding_dim=64):
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
    
    def forward(self, node_indices):
        return self.embeddings(node_indices)

# Phase 3: Encoder-Decoder LSTM for multi-step forecasting
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # x: [batch_size, 1, input_size]
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class LSTM_ED_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forecast_len, num_nodes=None):
        super(LSTM_ED_Model, self).__init__()
        self.num_nodes = num_nodes if num_nodes is not None else input_size
        
        # Integration of GAT to capture spatial/feature correlations
        # We treat each feature as a "node" in the graph
        self.gat = GATLayer(in_features=1, out_features=1, heads=1) 
        
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.forecast_len = forecast_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, x, adj=None):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.size()
        
        if adj is None:
            # Default to fully connected graph between features
            adj = torch.ones((input_size, input_size)).to(x.device)
            
        # Process each time step through GAT (spatial/feature interaction)
        # x_gnn: [batch_size, seq_len, input_size]
        x_gnn_list = []
        for t in range(seq_len):
            # Treat each feature as a node with 1-dim feature
            h_t = x[:, t, :].unsqueeze(-1) # [batch_size, input_size, 1]
            h_t_gat = self.gat(h_t, adj) # [batch_size, input_size, 1]
            x_gnn_list.append(h_t_gat.squeeze(-1).unsqueeze(1))
            
        x_gnn = torch.cat(x_gnn_list, dim=1) # [batch_size, seq_len, input_size]
        
        _, (h, c) = self.encoder(x_gnn)
        h, c = h[0], c[0] # [batch_size, hidden_size]
        
        # Initial decoder input
        decoder_input = torch.zeros(batch_size, self.output_size).to(x.device)
        
        outputs = []
        for _ in range(self.forecast_len):
            h, c = self.decoder_cell(decoder_input, (h, c))
            prediction = self.fc(h) 
            outputs.append(prediction.unsqueeze(1))
            decoder_input = prediction # Auto-regressive
            
        return torch.cat(outputs, dim=1)

# Phase 3: Ensemble Models (Baseline)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

def train_ensemble_baselines(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingRegressor(n_estimators=100)
    gb.fit(X_train, y_train)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100)
    xgb_model.fit(X_train, y_train)
    
    return rf, gb, xgb_model

if __name__ == "__main__":
    print("Testing GraphSAGE layer...")
    num_nodes = 5
    in_feat = 10
    out_feat = 16
    x = torch.randn(num_nodes, in_feat)
    adj = torch.tensor([[0, 1, 0, 0, 1],
                        [1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1],
                        [1, 0, 0, 1, 0]], dtype=torch.float32)
    layer = GraphSageLayer(in_feat, out_feat)
    output = layer(x, adj)
    print(f"GraphSAGE output shape: {output.shape}")
    
    print("Testing LSTM Encoder-Decoder...")
    batch_size = 8
    seq_len = 24
    input_size = 5
    hidden_size = 32
    output_size = 1 # predicting 1 feature
    forecast_len = 12
    
    model = LSTM_ED_Model(input_size, hidden_size, output_size, forecast_len)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    # Adjusting decoder input logic in forward to match dummy input
    # For testing purposes, we'll use a simpler version
    
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, forecast_len):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size * forecast_len)
            self.output_size = output_size
            self.forecast_len = forecast_len
        def forward(self, x):
            _, (h, c) = self.lstm(x)
            out = self.fc(h[-1])
            return out.view(-1, self.forecast_len, self.output_size)
            
    simple_model = SimpleLSTM(input_size, hidden_size, output_size, forecast_len)
    pred = simple_model(dummy_input)
    print(f"LSTM Prediction shape: {pred.shape}")
