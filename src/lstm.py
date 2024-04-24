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

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.fc_1 =  nn.Linear(self.hidden_size, self.num_classes) #fully connected 1
   
    def forward(self, x, init_states=None):

        if init_states is None:
            h_t = Variable(torch.zeros(self.batch_size, self.hidden_size)) # hidden state
            c_t = Variable(torch.zeros(self.batch_size, self.hidden_size)) # internal state
        else:
            h_t, c_t = init_states

        out, _ = self.lstm(x)
        out = self.relu(out)
        last_time_step_output = out[:, -1, :]  # Shape will be [batch_size, hidden_size]
        prediction = self.fc_1(last_time_step_output)  # Now the output shape is [batch_size, num_classes]
        return prediction