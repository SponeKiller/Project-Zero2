from torch import nn
import torch

class PositionalEmbedding(nn.Module):
    
    """
    Positional Embedding
    """
    
    def __init__(self, d_model: int) -> None:
        """
        Initialize Positional Embedding
        
        Args:
            d_model (int): Model dimension
        """
        
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Positional Embedding
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            x + pos_emb (torch.Tensor): Output tensor
            
        """
    
        
        pos_emb = self.pos_embedding(x)
        
        return x + pos_emb