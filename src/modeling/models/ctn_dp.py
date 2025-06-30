import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPMultiheadAttention

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):   
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = DPMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights with smaller values to reduce gradient magnitudes
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize linear layers with smaller weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of CustomTransformerEncoderLayers
    '''
    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)
        
        encode_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.h, 
            dim_feedforward=self.d_ff, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)
            
    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.mean(0) # global pooling
        return out

# 15 second model
class CTN(nn.Module):
    def __init__(self, in_channel, out_channel, d_model=256, nhead=8, d_ff=2048, num_layers=8, dropout_rate=0.2, deepfeat_sz=64):
        super(CTN, self).__init__()
        
        self.encoder = nn.Sequential( # downsampling factor = 20
            nn.Conv1d(in_channel, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.GroupNorm(32, 128),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.GroupNorm(32, 256),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.GroupNorm(32, d_model),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.GroupNorm(32, d_model),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, d_model),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, d_model),  # Replaced BatchNorm1d with GroupNorm
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.fc2 = nn.Linear(deepfeat_sz, out_channel)
        self.dropout = nn.Dropout(dropout_rate)
            
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(_weights_init)
            
    def forward(self, x):
        z = self.encoder(x)
        out = self.transformer(z)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out