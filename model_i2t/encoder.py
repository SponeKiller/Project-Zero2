from torch import nn
import torch

from model_i2t.residual import ResidualConnection
from model_i2t.attention import MultiHeadAttention
from model_i2t.mlp import MLP

class EncoderBlock(nn.Module):
    """
    Encoder Block
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 hidden_dim: int,
                 layers: int = 2, 
                 dropout: float = 0.3) -> None:
        """
        Inicialize Encoder Block
        
        Args:
            d_model (int): input dimension
            num_heads (int): number of heads
            hidden_dim (int): hidden dimension (default: 4 x d_model)
            layers (int): number of layers of MLP block
            dropout (float): dropout rate
        """      
        super().__init__()
        
        self.d_model = d_model
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, hidden_dim, layers, dropout)
        self.residual = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Foward pass Encoder Block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
            
        """
        
        for layer in self.residual:
            x = layer(x, lambda x: self.attention(x))
            x = layer(x, lambda x: self.mlp(x))	         
        
        return x

class Encoder(nn.Module):
    """
    Encoder
    
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 hidden_dim: int,
                 layers: int = 2,
                 enc_layers: int = 6,
                 dropout: float = 0.3) -> None:
        """
        Inicialize Encoder

        Args:
            d_model (int): input dimension
            num_heads (int): number of heads in MultiHeadAttention
            hidden_dim (int): hidden dimension in MLP
            layers (int, optional): number of layers in MLP. Defaults to 2.
            enc_layers (int, optional): number of layers in Encoder. 
                                        Defaults to 6.
            dropout (float, optional): dropout rate. Defaults to 0.3.
        """
        super().__init__()
        self.encoder = nn.ModuleList(
            [
            EncoderBlock(
                d_model,
                num_heads,
                hidden_dim,
                layers,
                dropout) for _ in range(enc_layers)
            ]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Foward pass Encoder
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        
        for layer in self.encoder:
            x = layer(x)
        
        return x