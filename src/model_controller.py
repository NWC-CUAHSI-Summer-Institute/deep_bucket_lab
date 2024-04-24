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
        self.input_vars = config['input_vars']
        self.output_vars = config['output_vars']
        self.model_config = config['model']
        self.seq_length = config['model']['seq_length']
        torch.manual_seed(1)
        self.lstm = deep_bucket_model(config['model']).to(device)
        self.scaler_in, self.scaler_out = self.fit_scaler()

    def fit_scaler(self):
        # Assuming scaler is fitted using data from all training buckets
        all_train_data = self.bucket_dictionary["train"]
        df_in = all_train_data[self.input_vars]
        scaler_in = StandardScaler()
        scaler_in.fit(df_in)
        df_out = all_train_data[self.output_vars]
        scaler_out = StandardScaler()
        scaler_out.fit(df_out)
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out
        return scaler_in, scaler_out

    def make_data_loader(self, split):
        bucket_list = self.bucket_dictionary[split]['bucket_id'].unique()
        loader = {}
        for ibuc in bucket_list:
            df = self.bucket_dictionary[split][self.bucket_dictionary[split]['bucket_id'] == ibuc]
            
            if df.empty:
                continue  # Skip if no data for this bucket

            data_in = self.scaler_in.transform(df[self.input_vars])
            data_out = self.scaler_out.transform(df[self.output_vars])
            
            seq_length = self.lstm.seq_length
            np_seq_X = [data_in[i:i+seq_length] for i in range(len(data_in) - seq_length)]
            np_seq_y = [data_out[i] for i in range(seq_length, len(data_out))]

            if not np_seq_X or not np_seq_y:
                continue  # Skip if sequences are empty

            ds = TensorDataset(torch.Tensor(np_seq_X), torch.Tensor(np_seq_y))
            loader[ibuc] = DataLoader(ds, batch_size=self.lstm.batch_size, shuffle=True)

        return loader  # Return only the loader dictionary


    def train_model(self, train_loader):
        criterion = torch.nn.MSELoss()
        num_epochs = self.config['model']['num_epochs']
        results = {}

        for epoch in range(num_epochs):
            epoch_losses = []

            for ibuc, loader in train_loader.items():  # Assuming train_loader is a dictionary of bucket-specific loaders
                bucket_losses = []

                for data, targets in loader:
                    optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.config['model']['learning_rate']['start'])
                    optimizer.zero_grad()
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.lstm(data)
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                    bucket_losses.append(loss.item())

                avg_loss = sum(bucket_losses) / len(bucket_losses)
                epoch_losses.append(avg_loss)
                if ibuc not in results:
                    results[ibuc] = {"loss": []}
                results[ibuc]["loss"].append(avg_loss)
#                print(f'Epoch {epoch+1}, Bucket {ibuc}, Avg Loss: {avg_loss}')

            # Calculate average loss for the epoch across all buckets
            total_avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch+1}, Total Avg Loss: {total_avg_loss}')

        return self.lstm, results
