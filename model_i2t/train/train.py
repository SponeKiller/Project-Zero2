import os
import json
from typing import List, Literal, Optional, Tuple, TypedDict
from tqdm import tqdm
from pathlib import Path
import inspect
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from ..main import VisionTransformer
from config import TrainArgs
from train.datasets.pretrain_dataset import Train_Dataset




class Train():
    
    """
    Train model
    
    """

    def __init__(self, model:VisionTransformer, config:TrainArgs):
        
        """
        Inicialize model training
        
        Args:
            model (VisionTransformer): Model to train
            config (TrainArgs): Configuration for training model
        """
        
        self.model: VisionTransformer = model
        self.config: TrainArgs = config
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        ).to(self.device)
        


    def pretrain_training(self):
        
        train_dataloader, val_dataloader = self._set_dataset()

        initial_epoch = 0
        global_step = 0


        for epoch in range(initial_epoch, self.config.num_epochs):
            torch.cuda.empty_cache()
            
            self.model.train()
            self._load_model_state(epoch)
            batch_iterator: torch.Tensor = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            
            for batch in batch_iterator:

             

                output = self.model.forward(batch['decoder_input'].to(self.device))

                # Compare the output with the label
                # (B, seq_len)
                label: torch.Tensor = batch['label'].to(self.device) 

                # Compute the loss using a simple cross entrophy
                loss: torch.Tensor = self.loss_fn(
                    output.view(-1, self.config.num_classes),
                    label.view(-1)
                )


                # Backpropagate the loss
                loss.backward()

                # Update the weights
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

            
            # Run validation at the end of every epoch
            self._run_validation(
                self.model, 
                val_dataloader, 
                tokenizer_src, 
                tokenizer_tgt, 
                config['seq_len'], 
                device, 
                lambda msg: batch_iterator.write(msg), 
                global_step, writer
            )

            # Save the model at the end of every epoch
            model_filename = self.get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    def finetune_training(self):
    
        """
        Finetune training
        """       
        
        print("Currently not implemented")
        return
    
    def reinforce_training(self):
        """
        Reinforce learinig 
        """
        print("Currently not implemented")
        return

        
        
    def _set_dataset(self) -> Tuple[DataLoader, DataLoader]:

        #loading data
        self.ds_raw = self._load_dataset(self.config.train_data)

        # Splitting Val and train ds
        assert self.config.train_ds_size > 1, "Train_ds_size must be less or equal 1"
        
        if self.config.augment:
            self.augment_dataset()
        
        train_ds_size: int = self.config.train_ds_size * len(self.ds_raw)
        val_ds_size: int = len(self.ds_raw) - train_ds_size

        train_ds_raw = self.ds_raw
        
        #Split ds only if we want something use for validation 
        if val_ds_size > 0:
            train_ds_raw, val_ds_raw = random_split(self.ds_raw, [train_ds_size, val_ds_size])
    
    
        train_ds = self.options[self.selected_training[1]](train_ds_raw, self.tokenizer, self.model.params.max_seq_len)
        
        if val_ds_size > 0:
            val_ds = self.options[self.selected_training[1]](val_ds_raw, self.tokenizer, self.model.params.max_seq_len)

        train_dataloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        
        
        if val_ds_size > 0:
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
            
        
        print("Dataset has been successfully loaded")

        return train_dataloader, val_dataloader
    

    def _load_dataset(self):

        """
        Load dataset from csv file

        Args:
            path (str): Path to the directory containing csv files.
            
        Returns:
            List[str] - data from csv file.

        Raises:
            AssertionError: if in provided path wont find any jsonl files.
        """
        
        file_path = Path(self.config.train_data)

        assert file_path.is_file(), (
            f"No jsonl files found in directory {self.config.train_data}"
        )
        
        
        data = []
        
        caller_function = inspect.stack()[1].function
        
        with open(file_path, "r") as f:
            for line in f:
                #Checking if input data for fine tuning are in correct shape
                if(caller_function == "pretrain_training"):
                    assert len(line) == 1, (f"Pretraing data should have only 1 column, but provided {len(line)}")
                elif(caller_function == "finetune_training"):    
                    assert line["messages"][0]["role"] == "system" and line["messages"][1::2]["role"] == "user" and line["messages"][2::2]["role"] == "assistant", ("model only supports 'system', 'user' and 'assistant' roles,starting with 'system', then 'user' and alternating (u/a/u/a/u...)")
                elif(caller_function == "reinforce_training"):    
                    assert len(line) == 1, (f"Reinforce_learning data should have only 1 column, but provided {len(line)}")
                else:
                    raise AssertionError("Other training types are not supported")
                
                data.append(json.loads(line))
             
        return data
        
        
    def augment_dataset(self):
        
        """
        Augment dataset
        
        Note:
        
        User can choose to augment dataset or not   
        If user choose to augment dataset, Augmentation class will be called
        
        """
        print("Currently not implemented")
        return
        
        augment = Augmentation(self.ds_raw)
        self.ds_raw = augment.augment_ds      


    def _run_validation(self,
                       validation_ds: torch.Tensor,
                       num_examples: int = 10) -> None:
        
        self.model.eval()
        
        count = 0

        source_image = []
        expected = []
        predicted = []


        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                # (b, seq_len)
                decoder_input = batch["decoder_input"].to(self.device) 

                output_img = self.model.forward(batch['decoder_input'].to(self.device))

                target_img = batch["label"]
                
                correct = torch.argmax(output_img, dim=1) == target_img
                
                print(f"{f'TARGET: ':>12}{target_img}")
                print(f"{f'PREDICTED: ':>12}{output_img}")

                if count == num_examples:
                    break
        
        print(f"ACCURACY: {count}/{num_examples}")


    def _get_weights_file_path(self, epoch: Optional[str] = "") -> str:
        """
        Get the path to the model weights file
        
        Args:
            epoch (Optional[str]): Epoch number
        
        Returns:
            str: Path to the model weights file
        """
        model_filename = f"{self.config.model_name}{epoch}.pt"
        return str(Path('.') / self.config.model_path / model_filename)
    
    def _save_model_state(self, epoch: Optional[str] = "") -> None:
        """
        Save model state
        
        Args:
            epoch (Optional[str]): Epoch number
        """
        
        model_filename = self._get_weights_file_path(epoch)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_filename)
        
        print(f"Model saved at {model_filename}")
    
    def _delete_model_state(self, epoch: Optional[str] = "") -> None:
        """
        Delete model state
        
        Args:
            epoch (Optional[str]): Epoch number
        """
        
        model_filename = self._get_weights_file_path(epoch)
        
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print(f"Model deleted at {model_filename}")
        else:
            print(f"No model to delete at {model_filename}")
    
    def _rename_model_state(self, epoch: Optional[str] = "") -> None:
        """
        Change model state to default
        
        Args:
            epoch (Optional[str]): Epoch number
        """
        
        old_model_filename = self._get_weights_file_path(epoch)
        new_model_filename = self._get_weights_file_path()
        
        if os.path.exists(old_model_filename):
            os.rename(old_model_filename, new_model_filename)
            print(f"Model state changed at {old_model_filename}")
        else:
            print(f"No model to change at {old_model_filename}")


    def _load_model_state(self, epoch: Optional[str] = "") -> None:
        """
        Load model state
        """
    
        if self.config.preload:
            model_filename = self._get_weights_file_path(epoch)
        else:
            model_filename = None
            
        if model_filename:
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])       
        else:
            print('No model to preload')
