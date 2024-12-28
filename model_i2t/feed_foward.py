from torch import nn
import torch

from model_i2t.gelu import Gelu

class FeedForward(nn.Module):
        
    """ 
    Feed Forward 
    
    """
    
    def __init__(self,
                input_dim: int, 
                hidden_dim: int, 
                dropout: float = 0.3) -> None:
        
        """
        Inicialize feed forward layer
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): hidden layer dimension
            dropout (float): dropout rate (during training)
        """
        
        super().__init__()
        
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = Gelu()
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of feed forward layer
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.output(x)
        
        return x