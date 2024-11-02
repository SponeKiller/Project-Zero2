from torch import nn
import torch
from residual import ResidualConnection
from attention import MultiHeadAttention
from mlp import MLP

class EncoderBlock(nn.Module):
    """
    Encoder Block
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = MLP 
        self.residual = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    """
    Encoder
    """
    
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64, 3, 1, 1)
        self.enc2 = EncoderBlock(64, 128, 3, 1, 1)
        self.enc3 = EncoderBlock(128, 256, 3, 1, 1)
        self.enc4 = EncoderBlock(256, 512, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))