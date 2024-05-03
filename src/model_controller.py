import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from lstm import deep_bucket_model

class ModelController:
    def __init__(self, config, device, bucket_dictionary):
        self.device = device
        self.config = config
        self.bucket_dictionary = bucket_dictionary
        self.do_config()
        torch.manual_seed(1)
        self.initialize_model()
        self.scaler_in = None
        self.scaler_out = None
        self.fit_scalerz()

    def initialize_model(self):
        self.lstm = deep_bucket_model(self.model_config).to(self.device)

    def do_config(self):
        self.model_config = self.config['model']
        self.input_vars = self.config['input_vars']
        self.output_vars = self.config['output_vars']
        self.seq_length = self.config['model']['seq_length']
        
    def fit_scalerz(self):
        B = self.bucket_dictionary["train"]
        # Create one scaler for all inputs
        self.scaler_in = StandardScaler()
        self.scaler_in.fit(B[self.input_vars])

        # Create a dictionary of scalers for each output variable
        self.scaler_out = {}
        for var in self.output_vars:
            scaler = StandardScaler()
            scaler.fit(B[[var]])  # Note double brackets to keep the DataFrame structure
            self.scaler_out[var] = scaler

    def make_data_loader(self, split):
        bucket_list = self.bucket_dictionary[split]['bucket_id'].unique()
        loader = {}
        for ibuc in bucket_list:
            df = self.bucket_dictionary[split][self.bucket_dictionary[split]['bucket_id'] == ibuc]
            
            if df.empty:
                continue  # Skip if no data for this bucket

            # Use the input scaler for all input variables
            data_in = self.scaler_in.transform(df[self.input_vars])
            
            # Initialize a list to collect transformed outputs, transformed per sequence rather than per point
            data_out = np.column_stack([self.scaler_out[var].transform(df[[var]]) for var in self.output_vars])
            
            seq_length = self.lstm.seq_length
            n = len(data_in)
            
            # Create input sequences
            np_seq_X = np.array([data_in[i:i+seq_length] for i in range(n - seq_length)])
            
            # Create output sequences that match the input sequences in structure and time steps
            np_seq_y = np.array([data_out[i:i+seq_length] for i in range(n - seq_length)])

            if np_seq_X.size == 0 or np_seq_y.size == 0:
                continue  # Skip if sequences are empty

            ds = TensorDataset(torch.tensor(np_seq_X, dtype=torch.float32), torch.tensor(np_seq_y, dtype=torch.float32))
            loader[ibuc] = DataLoader(ds, batch_size=self.lstm.batch_size, shuffle=False)

        return loader

    def train_model(self, train_loader):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.lstm.parameters(), 
            lr=self.config['model']['learning_rate']['start']
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config['model']['learning_rate']['step_size'], 
            gamma=self.config['model']['learning_rate']['gamma']
        )

        num_epochs = self.config['model']['num_epochs']
        for epoch in range(num_epochs):
            epoch_losses = []
            for ibuc, loader in train_loader.items():
                bucket_losses = []
                for data, targets in loader:
                    optimizer.zero_grad()
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.lstm(data)
                    targets = targets[:, -1, :] 
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                    bucket_losses.append(loss.item())

                avg_loss = sum(bucket_losses) / len(bucket_losses)
                epoch_losses.append(avg_loss)

            scheduler.step()
            total_avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch+1}: Total Avg Loss: {total_avg_loss}')

        return self.lstm