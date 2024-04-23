#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from lstm import deep_bucket_model

class ModelController:
    def __init__(self, config, device, bucket_dictionary):
        self.device = device
        self.config = config
        self.bucket_dictionary = bucket_dictionary
        
        model_config = config['model']
        torch.manual_seed(1)
        self.lstm = deep_bucket_model(model_config).to(device)
        
        # Assume that train_start and train_end are specified under each dataset type (train, val, test)
        self.scaler_in, self.scaler_out = self.fit_scaler()

    def fit_scaler(self):
        train_data = self.bucket_dictionary['train']
        input_vars = ['precip', 'et'] + list(train_data.columns.difference(['precip', 
                                                                            'et', 
                                                                            'q_overflow', 
                                                                            'q_spigot', 
                                                                            'bucket_id', 
                                                                            'time']))
        output_vars = ['q_overflow', 'q_spigot']
        
        df_in = train_data.loc[:, input_vars]
        scaler_in = StandardScaler()
        scaler_in.fit(df_in)
        
        df_out = train_data.loc[:, output_vars]
        scaler_out = StandardScaler()
        scaler_out.fit(df_out)
        
        return scaler_in, scaler_out

    def make_data_loader(self, bucket_key):
        df = self.bucket_dictionary[bucket_key]
        # Ensure all expected columns are considered
        input_vars = ['precip', 'et'] + list(df.columns.difference(['precip', 'et', 'q_overflow', 'q_spigot', 'bucket_id', 'time']))

        print("Input variables used:", input_vars)  # Debug: Print to check what's included

        if len(input_vars) != self.lstm.input_size:
            raise ValueError(f"Expected input_size {self.lstm.input_size}, but got {len(input_vars)} features.")

        output_vars = ['q_overflow', 'q_spigot']
        data_in = self.scaler_in.transform(df[input_vars])
        data_out = self.scaler_out.transform(df[output_vars])

        seq_length = self.lstm.seq_length
        np_seq_X = np.array([data_in[i:i+seq_length] for i in range(len(data_in) - seq_length)])
        np_seq_y = np.array([data_out[i] for i in range(seq_length, len(data_out))])

        if len(np_seq_X) == 0 or len(np_seq_y) == 0:
            raise ValueError("Generated sequences are empty. Check sequence generation logic.")

        ds = torch.utils.data.TensorDataset(torch.Tensor(np_seq_X), torch.Tensor(np_seq_y))
        loader = torch.utils.data.DataLoader(ds, batch_size=self.lstm.batch_size, shuffle=True)
        return loader


    def train_model(self, train_loader):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.config['model']['learning_rate']['start'])
        num_epochs = self.config['model']['num_epochs']
        
        for epoch in range(num_epochs):
            for data, targets in train_loader:
                optimizer.zero_grad()
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.lstm(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        return self.lstm
