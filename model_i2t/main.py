from torch import nn
import torch

from model_i2t.encoder import Encoder
from model_i2t.classification import Classification
from model_i2t.pos_embedding import PositionalEmbedding

class VisionTransformer(nn.Module):
    
    """
    Vision Transformer
    """
    
    def __init__(self,
                 patch_size: int, 
                 num_channels: int,
                 class_type: str,
                 num_classes: int,
                 d_model: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_layers: int = 2,
                 enc_layers: int = 6,  
                 dropout: float = 0.3):
        
        """
        
         Inicialize the Vision Transformer model
         
         Args:                
            patch_size: int
                The size of the patch
                
            num_channels: int
                The number of channels of the image
                
            class_type: str
                The type of classification
                
            num_classes: int
                The number of classes
                
            d_model: int
                The dimension of the model
                
            num_heads: int
                The number of heads
                
            hidden_dim: int
                The hidden dimension
                
            mlp_layers: int
                The number of layers of the mlp
                
            enc_layers: int
                The number of layers of the encoder
                
            dropout: float
                The dropout rate
        """
        
        #
        # HERE TO DO DATA VALIDATION
        #
        super().__init__()
    
        
        
        self.patch_embedding = nn.Conv2d(in_channels=num_channels,
                                         out_channels=d_model,            kernel_size=patch_size)
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.pos_embedding = PositionalEmbedding(d_model)
        
        self.encoder = Encoder(d_model,
                               num_heads,
                               hidden_dim,
                               mlp_layers,
                               enc_layers,
                               dropout)
        
        self.classification = Classification(class_type,
                                             d_model,
                                             num_classes)
                                             
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        
        Forward pass of the model
        
        Args:
            x: torch.Tensor
                The input tensor
                
        Returns:
            x: torch.Tensor
                The output tensor
        """
        
        ##
        # HERE TO DO DATA VALIDATION
        ##
        
          
    
        x = self.patch_embedding(x)
        
        x = self.pooling(x)
        
        #b, c, h, w -> b, p, c
        x = x.view(x.shape[0], -1, x.shape[1])
        
        x = self.pos_embedding(x)
        
        x = self.encoder(x)
        
        x = self.classification(x)
        
        return x