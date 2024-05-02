#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import torch.nn as nn
from torch.autograd import Variable

class deep_bucket_model(nn.Module):
    def __init__(self, config):
        super(deep_bucket_model, self).__init__()
        self.num_classes = config['num_classes']
        self.num_layers = config['num_layers']  # Make sure this is set correctly in your config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_state_size']
        self.seq_length = config['seq_length']
        self.batch_size = config['batch_size']

        # Initialize LSTM
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(self.hidden_size, self.num_classes) 

    def forward(self, x, init_states=None):
        if init_states is None:
            # Initialize hidden and cell states
            h_t = Variable(torch.zeros(1, x.size(0), self.hidden_size, device=x.device))
            c_t = Variable(torch.zeros(1, x.size(0), self.hidden_size, device=x.device))
            init_states = (h_t, c_t)

        # Process input through LSTM
        out, _ = self.lstm(x, init_states)
        out = self.relu(out[:, -1, :])  # Apply ReLU to last timestep
        prediction = self.fc_1(out)  # Generate prediction
        return prediction