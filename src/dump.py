#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable 
import sklearn
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import trange, tqdm
g = 9.807
time_step = 1
rain_probability_range = {"None": [0.6, 0.7], 
                          "Light": [0.5, 0.8], 
                          "Heavy": [0.2, 0.3]}

rain_depth_range = {"Light": [0, 2], "Heavy": [2, 8]}
bucket_attributes_range = {"A_bucket": [1.0, 2.0],
                           "A_spigot": [0.1, 0.2],
                           "H_bucket": [5.0, 6.0],
                           "H_spigot": [1.0, 3.0],
                           "K_infiltration": [1e-7, 1e-9],
                           "ET_parameter": [7, 9]
                          }

bucket_attributes_list = list(bucket_attributes_range.keys())
input_vars = ['precip', 'et']
input_vars.extend(bucket_attributes_list)
output_vars = ['q_overflow', 'q_spigot']
n_input = len(input_vars)
n_output = len(output_vars)
is_noise = True
noise = {"pet": 0.1, "et": 0.1, "q": 0.1, "head": 0.1} 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device: ", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
hidden_state_size = 16
num_layers = 8
num_epochs = 5  
batch_size = 256 
seq_length = 24
learning_rate = np.linspace(start=0.1, stop=0.01, num=num_epochs)
n_buckets_split = {"train": 10, "val": 5,"test": 1}
time_splits = {"train": 1000, "val": 500,"test": 500}
num_records = time_splits["train"] + time_splits["val"] + time_splits["test"] + seq_length * 3
n_buckets = n_buckets_split["train"] + n_buckets_split["val"] + n_buckets_split["test"]
def split_parameters():
    # create lists of bucket indices for each set based on the given bucket splits
    buckets_for_training = list(range(0, n_buckets_split['train'] + 1))
    buckets_for_val = list(range(n_buckets_split['train'] + 1, 
                                 n_buckets_split['train'] + n_buckets_split['val'] + 1))
    buckets_for_test = list(range(n_buckets - n_buckets_split['test'], n_buckets))

    # determine the time range for each set based on the given time splits
    train_start = seq_length
    train_end   = time_splits["train"]
    val_start   = train_end + seq_length
    val_end     = val_start + time_splits["val"]
    test_start  = val_end + seq_length
    test_end    = test_start + time_splits["test"]
    
    # organize the split parameters into separate lists for each set
    train_split_parameters = [buckets_for_training, train_start, train_end]
    val_split_parameters = [buckets_for_val, val_start, val_end]
    test_split_parameters = [buckets_for_test, test_start, test_end]
    
    return [train_split_parameters, val_split_parameters, test_split_parameters]
[[buckets_for_training, train_start, train_end],
[buckets_for_val, val_start, val_end],
[buckets_for_test, test_start, test_end]]= split_parameters()
def setup_buckets(n_buckets):
    # Boundary conditions
    buckets = {bucket_attribute:[] for bucket_attribute in bucket_attributes_list}
    for i in range(n_buckets):
        for attribute in bucket_attributes_list:
            buckets[attribute].append(np.random.uniform(bucket_attributes_range[attribute][0], 
                                                        bucket_attributes_range[attribute][1]))

    # Initial conditions
    h_water_level = [np.random.random() for i in range(n_buckets)]
    mass_overflow = [np.random.random() for i in range(n_buckets)]
    return buckets, h_water_level, mass_overflow

buckets, h_water_level, mass_overflow = setup_buckets(n_buckets)

def pick_rain_params():
    buck_rain_params = [rain_depth_range,
                        np.random.uniform(rain_probability_range["None"][0],
                                            rain_probability_range["None"][1]),
                        np.random.uniform(rain_probability_range["Heavy"][0],
                                            rain_probability_range["Heavy"][1]),
                        np.random.uniform(rain_probability_range["Light"][0],
                                            rain_probability_range["Light"][1])
                 ]
    return buck_rain_params

def random_rain(preceding_rain, bucket_rain_params):
    depth_range, no_rain_probability, light_rain_probability, heavy_rain_probability = bucket_rain_params
    # some percent of time we have no rain at all
    if np.random.uniform(0.01, 0.99) < no_rain_probability:
        rain = 0

    # When we do have rain, the probability of heavy or light rain depends on the previous day's rainfall
    else:
        # If yesterday was a light rainy day, or no rain, then we are likely to have light rain today
        if preceding_rain < depth_range["Light"][1]:
            if np.random.uniform(0, 1) < light_rain_probability:
                rain = np.random.uniform(0, 1)
            else:
                # But if we do have heavy rain, then it could be very heavy
                rain = np.random.uniform(depth_range["Heavy"][0], depth_range["Heavy"][1])

        # If it was heavy rain yesterday, then we might have heavy rain again today
        else:
            if np.random.uniform(0, 1) < heavy_rain_probability:
                rain = np.random.uniform(0, 1)
            else:
                rain = np.random.uniform(depth_range["Light"][0], depth_range["Light"][1])
    return rain

in_list = {}
for ibuc in range(n_buckets):
    bucket_rain_params = pick_rain_params()
    in_list[ibuc] = [0]
    for i in range(1, num_records):
        in_list[ibuc].append(random_rain(in_list[ibuc][i-1], bucket_rain_params))

def run_bucket_simulation(ibuc):
    columns = ['precip', 'et', 'h_bucket', 'q_overflow', 'q_spigot']
    columns.extend(bucket_attributes_list)
    # Memory to store model results
    df = pd.DataFrame(index=list(range(len(in_list[ibuc]))), columns=columns)
    
    # Main loop through time
    for t, precip_in in enumerate(in_list[ibuc]):
        
        # Add the input mass to the bucket
        h_water_level[ibuc] = h_water_level[ibuc] + precip_in

        # Lose mass out of the bucket. Some periodic type loss, evaporation, and some infiltration...
        et = np.max([0, (buckets["A_bucket"][ibuc] / buckets["ET_parameter"][ibuc]) * np.sin(t) * np.random.normal(1, noise['pet'])])
        infiltration = h_water_level[ibuc] * buckets["K_infiltration"][ibuc]
        h_water_level[ibuc] = np.max([0 , (h_water_level[ibuc] - et)])
        h_water_level[ibuc] = np.max([0 , (h_water_level[ibuc] - infiltration)])
        if is_noise:
            h_water_level[ibuc] = h_water_level[ibuc] * np.random.normal(1, noise['et'])

        # Overflow if the bucket is too full
        if h_water_level[ibuc] > buckets["H_bucket"][ibuc]:
            mass_overflow[ibuc] = h_water_level[ibuc] - buckets["H_bucket"][ibuc]
            h_water_level[ibuc] = buckets["H_bucket"][ibuc] 
            if is_noise:
                h_water_level[ibuc] = h_water_level[ibuc] - np.random.uniform(0, noise['q'])

        # Calculate head on the spigot
        h_head_over_spigot = (h_water_level[ibuc] - buckets["H_spigot"][ibuc] ) 
        if is_noise:
            h_head_over_spigot = h_head_over_spigot * np.random.normal(1, noise['head'])

        # Calculate water leaving bucket through spigot
        if h_head_over_spigot > 0:
            velocity_out = np.sqrt(2 * g * h_head_over_spigot)
            spigot_out = velocity_out *  buckets["A_spigot"][ibuc] * time_step
            h_water_level[ibuc] = h_water_level[ibuc] - spigot_out
        else:
            spigot_out = 0

        # Save the data in time series
        df.loc[t,'precip'] = precip_in
        df.loc[t,'et'] = et
        df.loc[t,'h_bucket'] = h_water_level[ibuc]
        df.loc[t,'q_overflow'] = mass_overflow[ibuc]
        df.loc[t,'q_spigot'] = spigot_out
        for attribute in bucket_attributes_list:
            df.loc[t, attribute] = buckets[attribute][ibuc]

        mass_overflow[ibuc] = 0
        
    return df

bucket_dictionary = {}
for ibuc in range(n_buckets):
    bucket_dictionary[ibuc] = run_bucket_simulation(ibuc)

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, batch_size, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length 
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc_1 =  nn.Linear(hidden_size, num_classes) #fully connected 1
   
    def forward(self, x, init_states=None):

        if init_states is None:
            h_t = Variable(torch.zeros(batch_size, self.hidden_size)) # hidden state
            c_t = Variable(torch.zeros(batch_size, self.hidden_size)) # internal state
        else:
            h_t, c_t = init_states
           
        out, _ = self.lstm(x)
        out = self.relu(out)
        prediction = self.fc_1(out) # Dense, fully connected layer
        
        return prediction

def check_validation_period(lstm, np_val_seq_X, ibuc, n_plot=100):
    
    def __make_prediction():
        lstm_output_val = lstm(torch.Tensor(np_val_seq_X[ibuc]).to(device=device))
        val_spigot_prediction = []
        val_overflow_prediction = []
        for i in range(lstm_output_val.shape[0]):
            val_spigot_prediction.append((lstm_output_val[i,-1,1].cpu().detach().numpy() * \
                                    np.std(df.loc[train_start:train_end,'q_spigot'])) + \
                                   np.mean(df.loc[train_start:train_end,'q_spigot']))

            val_overflow_prediction.append((lstm_output_val[i,-1,0].cpu().detach().numpy() * \
                                    np.std(df.loc[train_start:train_end,'q_overflow'])) + \
                                   np.mean(df.loc[train_start:train_end,'q_overflow']))
        return val_spigot_prediction, val_overflow_prediction
    
    def __compute_nse():
        spigot_out = df.loc[val_start:val_end, 'q_spigot']
        spigot_mean = np.mean(spigot_out)
        spigot_pred_variance = 0
        spigot_obs_variance = 0

        overflow_out = df.loc[val_start:val_end, 'q_overflow']
        overflow_mean = np.mean(overflow_out)
        overflow_pred_variance = 0
        overflow_obs_variance = 0

        for i, pred_spigot in enumerate(val_spigot_prediction):
            t = i + seq_length - 1
            spigot_pred_variance += np.power(( pred_spigot          - spigot_out.values[t]), 2)
            spigot_obs_variance  += np.power(( spigot_mean          - spigot_out.values[t]), 2)

        for i, pred_overflow in enumerate(val_overflow_prediction):
            t = i + seq_length - 1
            overflow_pred_variance += np.power((pred_overflow          - overflow_out.values[t]), 2)
            overflow_obs_variance  += np.power((overflow_mean          - overflow_out.values[t]), 2)
        spigot_nse = np.round( 1 - ( spigot_pred_variance / spigot_obs_variance   ), 4)
        overland_flow_nse = np.round( 1 - ( overflow_pred_variance / overflow_obs_variance ), 4)
        return spigot_nse, overland_flow_nse

    def __compute_mass_balance():
        mass_in = df.sum()['precip']
        mass_out = df.sum()['et'] + \
                   df.sum()['q_overflow'] + \
                   df.sum()['q_spigot'] + \
                   df.loc[num_records - 1, 'h_bucket']
        return mass_in, mass_out

        
    df = bucket_dictionary[ibuc]
    val_spigot_prediction, val_overflow_prediction = __make_prediction()
    spigot_nse, overland_flow_nse = __compute_nse()
    mass_in, mass_out = __compute_mass_balance()
        
    print("Spigot NSE", spigot_nse)
    print("Overflow NSE", overland_flow_nse)
    print("Mass into the system: ", mass_in)
    print("Mass out or left over:", mass_out)
    print("percent mass resudual: {:.0%}".format((mass_in - mass_out) /mass_in))

torch.manual_seed(1)
lstm = LSTM1(num_classes=n_output,  
             input_size=n_input,    
             hidden_size=hidden_state_size, 
             num_layers=num_layers, 
             batch_size=batch_size, 
             seq_length=seq_length).to(device=device)

def fit_scaler():
    frames = [bucket_dictionary[ibuc].loc[train_start:train_end, input_vars] for ibuc in buckets_for_training]
    df_in = pd.concat(frames)    
    scaler_in = StandardScaler()
    scaler_train_in = scaler_in.fit_transform(df_in)

    frames = [bucket_dictionary[ibuc].loc[train_start:train_end, output_vars] for ibuc in buckets_for_training]
    df_out = pd.concat(frames)    
    scaler_out = StandardScaler()
    scaler_train_out = scaler_out.fit_transform(df_out)
    return scaler_in, scaler_out

scaler_in, scaler_out = fit_scaler()

def make_data_loader(start, end, bucket_list):
    loader = {}
    np_seq_X = {}
    np_seq_y = {}
    for ibuc in bucket_list:
        df = bucket_dictionary[ibuc]
        scaler_in_i = scaler_in.transform(df.loc[start:end, input_vars])
        scaler_out_i = scaler_out.transform(df.loc[start:end, output_vars])
        np_seq_X[ibuc] = np.zeros((scaler_in_i.shape[0] - seq_length, seq_length, n_input))
        np_seq_y[ibuc] = np.zeros((scaler_out_i.shape[0] - seq_length, seq_length, n_output))
        for i in range(0, scaler_in_i.shape[0] - seq_length):
            t = i+seq_length
            np_seq_X[ibuc][i, :, :] = scaler_in_i[i:t,:]
            np_seq_y[ibuc][i, :, :] = scaler_out_i[i:t,:]

        ds = torch.utils.data.TensorDataset(torch.Tensor(np_seq_X[ibuc]), 
                                                  torch.Tensor(np_seq_y[ibuc]))
        loader[ibuc] = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, np_seq_X, np_seq_y

train_loader, np_train_seq_X, np_train_seq_y = make_data_loader(train_start, train_end, buckets_for_training)
val_loader, np_val_seq_X, np_val_seq_y = make_data_loader(val_start, val_end, buckets_for_val)
test_loader, np_test_seq_X, np_test_seq_y = make_data_loader(test_start, test_end, buckets_for_test)

def train_model(lstm, train_loader, buckets_for_training):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate[0])
    epoch_bar = tqdm(range(num_epochs),desc="Training", position=0, total=num_epochs)
    
    # Create a dictionary to store the results
    results = {}
    for epoch in epoch_bar:

        for ibuc in buckets_for_training:

            batch_bar = tqdm(enumerate(train_loader[ibuc]),
                             desc="Bucket: {}, Epoch: {}".format(str(ibuc),str(epoch)),
                             position=1,
                             total=len(train_loader[ibuc]), leave=False, disable=True)

            for i, (data, targets) in batch_bar:

                optimizer.zero_grad()

                optimizer = optim.Adam(lstm.parameters(), lr=learning_rate[epoch])

                data = data.to(device=device)
                targets = targets.to(device=device)

                # Forward
                lstm_output = lstm(data) 
                loss = criterion(lstm_output, targets)

                #backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()

                batch_bar.set_postfix(loss=loss.to(device).item(),
                                      RMSE="{:.2f}".format(loss**(1/2)),
                                      epoch=epoch)
                batch_bar.update()

            with torch.no_grad():
                rmse_list = []
                for i, (data_, targets_) in enumerate(train_loader[ibuc]):
                    data_ = data_.to(device=device)
                    targets_ = targets_.to(device=device)
                    lstm_output_ = lstm(data_)
                    MSE_ = criterion(lstm_output_, targets_)
                    rmse_list.append(MSE_**(1/2))

            meanrmse = np.mean(np.array(torch.Tensor(rmse_list)))
            epoch_bar.set_postfix(loss=loss.cpu().item(),
                                  RMSE="{:.2f}".format(meanrmse),
                                  epoch=epoch)
            
            if ibuc not in results:
                results[ibuc] = {"loss": [], "RMSE": []}
            results[ibuc]["loss"].append(loss.cpu().item())
            results[ibuc]["RMSE"].append(meanrmse)
            #....todo..? also add IT metrics in results in this function ...
            batch_bar.update()
        
    return lstm, results

lstm, results = train_model(lstm, train_loader, buckets_for_training)

for ibuc in buckets_for_val:
    check_validation_period(lstm, np_val_seq_X, ibuc)