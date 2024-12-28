from torch import nn
import torch

from model_i2t.feed_foward import FeedForward

class MLP(nn.Module):
    """
    Multi layer perceptron
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 layers: int = 2, 
                 dropout: float = 0.3) -> None:
        """
        Inicialize MLP
        
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            layers (int): number of layers
            dropout (float): dropout rate
        
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        self.mlp = nn.ModuleList([FeedForward(input_dim, hidden_dim, dropout) for _ in range(layers)])

    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Foward pass MLP
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
            
        Assertions:
            Input size should be equal to d_model
            
        """
        assert self.input_dim == x.size(-1), (
            f"""Input tensor has shape {x.size(-1)},
            but expected {self.input_dim}"""
        )
        
        for layer in self.mlp:
            x = layer(x)

        return x