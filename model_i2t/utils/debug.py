from typing import Optional

import torch
from pathlib import Path
from datetime import datetime


from model_i2t.main import VisionTransformer

class Debug:
    """
    Debugging utilities
    """
    def __init__(
        self,
        num_epochs: int,
        file_path: str,
        enable: bool = True
    ):
        self.enable = enable
        self.num_epochs = num_epochs
        self.loss = []
        self.total_loss = 0
        self.accuracy = []
        self.total_accuracy = 0
        
        
        # Create file to write debug information 
        path = Path(file_path)
        
        file_name = path.stem
        file_suffix = path.suffix
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.file_path = f"{file_name}_{current_time}{file_suffix}"
        
        if path.is_dir():
            self.file_path = path.parent / self.file_path
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        
        
    def render_loss(self, 
                    loss: torch.Tensor, 
                    curr_epoch: int, 
                    avg_per_epochs: int = 1,
                    type: str = "train",):
        """
        Render the calculation graph of the model
        
        Args:
            model (VisionTransformer): The model
            tensor (torch.Tensor): The tensor to render the graph for
            name (str): The name of the file to save the graph to
        """
        if self.enable is False:
            return
  
        print(f"{type.upper()} LOSS: {loss}")
 
        self.loss.append(loss)
              
        if curr_epoch > avg_per_epochs:
            print(
                f"AVG {type.upper()} LOSS PER {avg_per_epochs}: "
                f"{sum(self.loss) / avg_per_epochs}"
            )
            
            self.loss.pop(0)
        
        self.total_loss += loss
        
        if self.num_epochs == curr_epoch:
            print(f"FINAL AVG {type.upper()} LOSS: "
                  f"{self.loss / self.num_epochs}"
            )
                
    def render_accuracy(self, 
                        prediction: torch.Tensor, 
                        labels: torch.Tensor,
                        curr_epoch: int, 
                        avg_per_epochs: int = 1,
                        type: str = "train",
                        show_prediction: bool = True,):

        if self.enable is False:
            return
        
        if show_prediction:
            print(f"{type.upper()} PREDICTION: {prediction}")
            print(f"{type.upper()} LABELS: {labels}")
        
        correct = (prediction == labels).sum().item()
        
        print(f"{type.upper()} ACCURACY: "
              f"{(correct / len(prediction)) * 100} %"
        )
        
        self.accuracy.append(correct / len(prediction))
        
        if curr_epoch > avg_per_epochs:
            print(
                f"AVG {type.upper()} ACCURACY PER {avg_per_epochs}: "
                f"{(sum(self.accuracy) / avg_per_epochs) * 100} %"
            )
            self.accuracy.pop(0)
            
        self.total_accuracy += correct / len(prediction)
        
        if self.num_epochs == curr_epoch:
            print(
                f"FINAL AVG {type.upper()} ACCURACY: "
                f"{(self.total_accuracy / self.num_epochs) * 100} %"
            )


    def write_to_file(self, message: str, heading: Optional[str] = None):
        """
        Write a message to the debug file
        
        Args:
            message (str): The message to write
        """
        if self.enable is False:
            return
        
        heading_len = len(heading) if heading is not None else 0
        
        with open(self.file_path, "a") as file:
            if heading is not None:
                file.write("+" + "-" * (heading_len + 2) + "+")
                file.write("\n")
                file.write("| " + heading + " |")
                file.write("\n")
                file.write("+" + "-" * (heading_len + 2) + "+")
                file.write("\n\n")
            
            file.write(message)
            file.write("\n")