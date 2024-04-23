#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import torch.nn as nn
from torch.autograd import Variable

class deep_bucket_model(nn.Module):
    def __init__(self, config):
        super(deep_bucket_model, self).__init__()
        self.num_classes = config['num_classes']
        self.num_layers = config['num_layers']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_state_size']
        self.seq_length = config['seq_length']
        self.batch_size = config['batch_size']

        # LSTM and fully connected layer initialization
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_size, self.num_classes)
   
    def forward(self, x, init_states=None):
        """
        Defines the forward pass of the LSTM model.
        """
        if init_states is None:
            # Initialize hidden and cell states with dimensions:
            # [num_layers, batch_size, hidden_size]
            h_t = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))  # hidden state
            c_t = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))  # internal state
        else:
            h_t, c_t = init_states
           
        out, _ = self.lstm(x, (h_t, c_t))
        out = out[:, -1, :]  # Get the outputs for the last time step
        prediction = self.fc_1(out)  # Apply the fully connected layer
        
        return prediction