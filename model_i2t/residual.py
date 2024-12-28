from torch import nn
import torch

from model_i2t.layer_norm import LayerNorm

class ResidualConnection(nn.Module):
    
    """
        Residual connection
    """
    
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.3) -> None:
    
        """
        Initializes residual connection 
        
        Args:
            d_model (int): Dimension of the model
            dropout (float): Dropout rate 
        
            
        """
            
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, 
                x: torch.Tensor, 
                sublayer: nn.Module) -> torch.Tensor:
        
        """
        Forward pass of residual connection
        
        Args:
            x (torch.Tensor): Input tensor
            sublayer (nn.Module): Sublayer
        returns:
            torch.Tensor: Output tensor
        """
        
        return x + sublayer(self.dropout(self.norm(x)))