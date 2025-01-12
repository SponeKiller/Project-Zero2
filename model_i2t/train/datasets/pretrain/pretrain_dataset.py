import torch
from torch.utils.data import Dataset

class Train_Dataset(Dataset):

    def __init__(self, ds, dtype=torch.float32):
        
        # Check if the dataset has the required keys
        for key in ["data", "labels"]:
            if key not in ds:
                raise Exception(
                    f"Key '{key}' is missing in the data!"
                )
            
        super().__init__()
        self.ds = ds
        self.dtype = dtype 

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        decoder_input = torch.tensor(self.ds["data"][idx], dtype=self.dtype)
        
        return {
            "decoder_input": decoder_input,
            "labels": self.ds["labels"][idx],
        }
