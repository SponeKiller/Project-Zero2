from torch import nn
import torch
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention
    
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: int = 0.3) -> None:
        
        """
        Initialize MultiHeadAttention
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of heads
            dropout (float): Dropout rate (during training)
        
        Assertions:
            d_model should be multiple of num_heads
        """
        
        assert (
            d_model % num_heads == 0
        ),"d_model should be multiple of num_heads"
        
        
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model / num_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    @staticmethod
    def attention(q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  dropout: nn.Dropout) -> torch.Tensor:
        """
        Calculate attention
        
        Args:
            q (torch.Tensor): Input tensor 
            k (torch.Tensor): Input tensor
            v (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask tensor
            dropout: nn.Dropout: Dropout layer
        
        Returns:
            attention: torch.Tensor
                 
        """
        d_k = q.size(-1)
        
        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) /  torch.sqrt(d_k)
        
        
        attention_scores = attention_scores.softmax(dim=-1)
        
        
        if dropout is not None:
            dropout(attention_scores)
        
        return torch.matmul(attention_scores, v)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of MultiHeadAttention
        
        Args:
            x (torch.Tensor): Input tensor
            mask Optional(torch.Tensor): Mask tensor
        
        Returns:
            torch.Tensor: Output tensor
        
        Assertions:
            Input size should be equal to d_model
        """
        
        assert (self.d_model == x.size(-1),
                "Input size should be equal to d_model")
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), int(self.num_heads), int(self.d_k)).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), int(self.num_heads), int(self.d_k)).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), int(self.num_heads), int(self.d_k)).transpose(1, 2)
        
        attention = MultiHeadAttention.attention(query, key, value, self.dropout)
        
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attention = attention.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_k * self.num_heads)
        
        
        return self.o(attention)