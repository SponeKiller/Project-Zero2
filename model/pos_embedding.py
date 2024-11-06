from torch import nn
import torch

class PosEmbedding(nn.Module):
    
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
            
        Assertions:
            Input dimension should be equal to d_model
        """
        
        
        assert self.d_model == x.size(-1), f"Input dimension {x.size(-1)} should be equal to d_model {self.d_model}"
        
        pos_emb = self.pos_embedding(x)
        
        return x + pos_emb