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
        self.labels = torch.tensor(self.ds["labels"], dtype=torch.long)
        self.decoder_input = torch.tensor(self.ds["data"], dtype=dtype)
        
        if(self.decoder_input.shape[0] != self.labels.shape[0]):
            raise ValueError("The number of samples in the data and labels should be the same")
        
        if (self.decoder_input.max() > 1 and self.decoder_input.min() == 0):
            # Normalize the data
            print("Normalizing the data to range [0, 1]")
            self.decoder_input = self.decoder_input / self.decoder_input.max()
        else:
            raise ValueError("Data should be normalized between 0 and 1")
        

    def __len__(self):
        return len(self.decoder_input)

    def __getitem__(self, idx):
        
        return {
            "decoder_input": self.decoder_input[idx],
            "labels": self.labels[idx],
        }
