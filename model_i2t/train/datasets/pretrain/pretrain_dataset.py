import torch
from torch.utils.data import Dataset

class Train_Dataset(Dataset):

    def __init__(self, ds):
        
        # Check if the dataset has the required keys
        for key in ["data", "label"]:
            if key not in ds:
                raise Exception(
                    f"Key '{key}' is missing in the data!"
                )
            
        super().__init__()
        self.ds = ds 

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return {
            "decoder_input": self.ds["data"][idx],
            "label": self.ds["label"][idx],
        }
