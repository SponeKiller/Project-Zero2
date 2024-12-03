import os
import json
from typing import List, Literal, Optional, Tuple, TypedDict
from tqdm import tqdm
from pathlib import Path

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
        


    def pretrain_training(self):
        
        train_dataloader, val_dataloader = self._set_dataset()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, eps=1e-9, weight_decay=self.config.weight_decay)

        initial_epoch = 0
        global_step = 0

        
        if self.config['preload'] == "latest":
            model_filename = self.latest_weights_file_path()
        elif self.config['preload']:
            model_filename = self.get_weights_file_path()
        else:
            model_filename = None
            
        if model_filename:
            
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            self.model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print('No model to preload, starting from scratch')
        
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

        for epoch in range(initial_epoch, self.config.num_epochs):
            torch.cuda.empty_cache()
            
            self.model.train()
            
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            
            for batch in batch_iterator:

             

                output = self.model.forward(batch['input_ids'].to(self.device))

                # Compare the output with the label
                # (B, seq_len)
                label: torch.Tensor = batch['label'].to(self.device) 

                # Compute the loss using a simple cross entropy
                loss = loss_fn(output.view(-1, self.config.num_classes), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            
            # Run validation at the end of every epoch
            self.run_validation(
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
    
    def reinforce_training(self):
        """
        Reinforce learinig 
        """
        
        
        
    def _set_dataset(self):

        #loading data
        self.ds_raw = self._load_dataset(self.config.train_data)

        # Splitting Val and train ds
        assert self.config.train_ds_size > 1, "Train_ds_size must be less or equal 1"
        
        #Checking if user wants to augment dataset
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
            AssertionError: if in provided path wont find any csv files.
            AssertionError: if columns wont be in correct order
            AssertionError: if columns is more than should be provided 
        """
        
        files = sorted(Path(self.config.train_data).glob("*.jsonl"))

        assert len(files) > 0, f"No jsonl files found in directory {self.config.train_data}"

        
        if (len(files) > 1):
            
            term = Terminal()
            
            with term.cbreak():
                # Starting index
                selected = 0

                print("Please select file for training model")
                
                # Event loop
                while True:
                    print(term.move_yx(0, 0) + term.clear)
                    
                    for index, option in enumerate(files):
                        if index == selected:
                            print(term.underline + option + term.no_underline)
                        else:
                            print(option)

                    inp = term.inkey()
                    
                    if inp.is_sequence:
                        if inp.name == "KEY_UP":
                            selected -= 1
                        elif inp.name == "KEY_DOWN":
                            selected += 1
                        elif inp.name == "KEY_ENTER":
                            break


                    # Stay within the options list
                    selected %= len(files)

            selected_file = files[selected]
            
        else:
            selected_file = files[0]
        
        data = []
        
        with open(selected_file, "r") as file:
            for line in file:
                #Checking if input data for fine tuning are in correct shape
                if(self.selected_training == "Pretraing"):
                    assert len(line) == 1, (f"Pretraing data should have only 1 column, but provided {len(line)}")
                ## Tady jeste orpavit
                if(self.selected_training == "Finetuning"):    
                    assert line["messages"][0]["role"] == "system" and line["messages"][1::2]["role"] == "user" and line["messages"][2::2]["role"] == "assistant", ("model only supports 'system', 'user' and 'assistant' roles,starting with 'system', then 'user' and alternating (u/a/u/a/u...)")
                    
                if(self.selected_training == "Reinforce_learning"):    
                    assert len(line) == 1, (f"Reinforce_learning data should have only 1 column, but provided {len(line)}")
                
                data.append(json.loads(line))
        
        
        
        print(f"Selected file: {selected_file} is loading.") 
             
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


    def run_validation(self, model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
        model.eval()
        count = 0

        source_texts = []
        expected = []
        predicted = []

        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be 1 for validation"

                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                
                # Print the source, target and model output
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg('-'*console_width)
                    break

    def get_weights_file_path(self, config, epoch: str):
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

    # Find the latest weights file in the weights folder
    def latest_weights_file_path(self, config):
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])
