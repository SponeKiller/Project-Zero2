from torchviz import make_dot
import torch

from model_i2t.main import VisionTransformer

class Debug:
    """
    Debugging utilities
    """
    
    @staticmethod
    def render_calc_graph(model: VisionTransformer,
                          tensor: torch.Tensor, 
                          name: str):
        """
        Render the calculation graph of the model
        
        Args:
            model (VisionTransformer): The model
            tensor (torch.Tensor): The tensor to render the graph for
            name (str): The name of the file to save the graph to
        """
        
        dot = make_dot(tensor, params=dict(model.named_parameters()))
        dot.format = 'svg'
        dot.render(name, cleanup=True)
        return dot