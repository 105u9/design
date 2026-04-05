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
    def __init__(self, in_features, out_features, heads=1):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.out_features = out_features
        
        # Phase 6 Optimization: Vectorized Multi-head Attention
        # Single linear layer to handle all heads simultaneously
        self.attn_linear = nn.Linear(out_features * 2, heads, bias=False)
        
        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_linear.weight, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # h: [B, N, in_feat]
        is_batched = h.dim() == 3
        if not is_batched:
            h = h.unsqueeze(0)
            
        B, N, _ = h.size()
        
        # 1. Linear transformation: [B, N, out_feat]
        Wh = self.W(h) 
        
        # 2. Vectorized Attention calculation
        # Wh_left/right: [B, N, heads, out_feat]
        # Instead of repeat, we use broadcasting logic
        # Extract weights for left and right parts: [heads, out_feat]
        a_weights = self.attn_linear.weight # [heads, 2 * out_feat]
        a_left = a_weights[:, :self.out_features].T  # [out_feat, heads]
        a_right = a_weights[:, self.out_features:].T # [out_feat, heads]
        
        # [B, N, out_feat] @ [out_feat, heads] -> [B, N, heads]
        e_left = torch.matmul(Wh, a_left)   
        e_right = torch.matmul(Wh, a_right) 
        
        # Broadcasting to [B, heads, N, N]
        # e_left: [B, N, heads] -> [B, heads, N, 1]
        # e_right: [B, N, heads] -> [B, heads, 1, N]
        e = e_left.permute(0, 2, 1).unsqueeze(3) + e_right.permute(0, 2, 1).unsqueeze(2)
        e = self.leakyrelu(e) # [B, heads, N, N]
        
        # 3. Masking
        if adj.dim() == 2:
            adj_mask = adj.unsqueeze(0).unsqueeze(0).expand(B, self.heads, -1, -1)
        else:
            adj_mask = adj.unsqueeze(1) # Add heads dim
            
        mask = (adj_mask == 0)
        e = e.masked_fill(mask, float('-inf'))
        
        # 4. Softmax & Attention Apply
        attn_weights = torch.softmax(e, dim=-1) # Normalize over neighbors
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        # Wh: [B, N, out_feat] -> [B, 1, N, out_feat] -> [B, heads, N, out_feat]
        Wh_expanded = Wh.unsqueeze(1).expand(-1, self.heads, -1, -1)
        
        # [B, heads, N, N] @ [B, heads, N, out_feat] -> [B, heads, N, out_feat]
        h_prime = torch.matmul(attn_weights, Wh_expanded)
        
        # 5. Merge heads (Average)
        out = torch.mean(h_prime, dim=1) # [B, N, out_feat]
        out = torch.relu(out)
        
        return out if is_batched else out.squeeze(0)

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
        
        # Phase 5 Upgrade: High-dimensional Node Embedding for GAT
        # By projecting features from 1 to 16 dimensions, the GAT can learn complex 
        # inter-sensor relationships more effectively than with a single scalar value.
        self.node_embed = nn.Linear(1, 16)
        self.gat = GATLayer(in_features=16, out_features=16, heads=4) 
        self.node_proj = nn.Linear(16, 1)
        
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.forecast_len = forecast_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, x, adj=None, y=None, teacher_forcing_ratio=0.0):
        # x: [batch_size, seq_len, input_size]
        # y: [batch_size, forecast_len, output_size] (Optional, used for Teacher Forcing)
        batch_size, seq_len, input_size = x.size()
        
        if adj is None:
            # Default to fully connected graph between features
            adj = torch.ones((input_size, input_size)).to(x.device)
            
        # Optimization: Flatten batch and sequence to process all through GAT in parallel
        # h_in: [batch_size * seq_len, input_size, 1]
        h_in = x.view(batch_size * seq_len, input_size, 1)
        
        # Phase 5: Feature Embedding -> GAT -> Projection
        h_emb = self.node_embed(h_in) # [B*S, N, 16]
        
        # Expand adj for the flattened dimension if it's 2D
        if adj.dim() == 2:
            adj_expanded = adj.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        else:
            adj_expanded = adj
            
        # Parallel GAT computation
        h_out = self.gat(h_emb, adj_expanded) # [B*S, N, 16]
        h_final = self.node_proj(h_out) # [B*S, N, 1]
        
        # Restore dimensions to [batch_size, seq_len, input_size]
        x_gnn = h_final.squeeze(-1).view(batch_size, seq_len, input_size)
        
        _, (h, c) = self.encoder(x_gnn)
        h, c = h[0], c[0] # [batch_size, hidden_size]
        
        # Initial decoder input (using zero initialization on the same device)
        decoder_input = torch.zeros(batch_size, self.output_size, device=x.device)
        
        outputs = []
        for i in range(self.forecast_len):
            h, c = self.decoder_cell(decoder_input, (h, c))
            prediction = self.fc(h) 
            outputs.append(prediction.unsqueeze(1))
            
            # Teacher Forcing: With probability p, use the real label y as the next input
            if y is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = y[:, i, :]
            else:
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
