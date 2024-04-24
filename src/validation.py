#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import numpy as np
import yaml
from vizualization import plot_timeseries

class ModelValidator:
    def __init__(self, lstm, 
                 device, 
                 bucket_dictionary, 
                 loader, config, split, scaler_out):
        """
        Initialize the ModelValidator with split-specific configuration.

        Args:
            lstm (torch.nn.Module): The LSTM model to be validated.
            device (torch.device): The device (GPU/CPU) the model is running on.
            bucket_dictionary (dict): Has the stuff
            loader (loader): has the other stuff
            config (dict): Configuration dictionary loaded from the YAML file.
            split (str): 'train', 'val', or 'test' to indicate the data split.
            scaler_out (sklearn.preprocessing.StandardScaler): Scaler used for output variables.
        """
        self.model = lstm
        self.device = device
        self.config = config['synthetic_data'][split]
        self.bucket_dictionary = bucket_dictionary
        self.train_start = self.config['num_records']  # Adjust based on actual config structure
        self.train_end = self.config['num_records']  # Adjust as needed
        self.seq_length = config['model']['seq_length']
        self.loader = loader
        self.split = split
        self.scaler_out = scaler_out

    def compute_nse(self, spigot_prediction, overflow_prediction, spigot_out, overflow_out):

        # Check lengths match
        if len(spigot_prediction) != len(spigot_out):
            raise ValueError(f"Length mismatch between predicted spigot output ({len(spigot_prediction)}) and actual spigot output ({len(spigot_out)})")

        if len(overflow_prediction) != len(overflow_out):
            raise ValueError(f"Length mismatch between predicted overflow output ({len(overflow_prediction)}) and actual overflow output ({len(overflow_out)})")

        # Calculate variances for NSE
        spigot_pred_variance = np.sum(np.power(np.array(spigot_prediction) - spigot_out, 2))
        spigot_obs_variance = np.sum(np.power(spigot_out.mean() - spigot_out, 2))
        overflow_pred_variance = np.sum(np.power(np.array(overflow_prediction) - overflow_out, 2))
        overflow_obs_variance = np.sum(np.power(overflow_out.mean() - overflow_out, 2))

        # Calculate NSE
        spigot_nse = 1 - (spigot_pred_variance / spigot_obs_variance) if spigot_obs_variance != 0 else float('nan')
        overland_flow_nse = 1 - (overflow_pred_variance / overflow_obs_variance) if overflow_obs_variance != 0 else float('nan')

        return spigot_nse, overland_flow_nse

    def compute_mass_balance(self, ibuc):
        df = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
        mass_in = df['precip'].sum()
        mass_out = df['et'].sum() + df['q_overflow'].sum() + df['q_spigot'].sum() + df.loc[df.index[-1], 'h_bucket']
        return mass_in, mass_out

    def post_process_predictions(self, model_outputs):
        # Implement any necessary transformation or extraction of predictions from model outputs
        # This is a placeholder function and needs actual implementation based on model output structure
        spigot_predictions = [output[0] for output in model_outputs]
        overflow_predictions = [output[1] for output in model_outputs]
        return spigot_predictions, overflow_predictions
    
    def validate_model(self, do_plot_timeseries=False):
        for ibuc in self.bucket_dictionary[self.split]['bucket_id'].unique():
            df = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
            
            # Assume the predictions start after `seq_length` due to the need for initial sequence context
            spigot_out = df['q_spigot'].iloc[self.seq_length:].to_numpy()
            overflow_out = df['q_overflow'].iloc[self.seq_length:].to_numpy()

            spigot_predictions = []
            overflow_predictions = []

            loader = self.loader[ibuc]  # DataLoader for the specific bucket
            for data, _ in loader:
                data, _ = data.to(self.device), _.to(self.device)
                output = self.model(data)
                print(output.shape)
                # Rescale predictions to original scale
                scaled_outputs = self.scaler_out.inverse_transform(output.detach().cpu().numpy())
                print(scaled_outputs.shape)
                spigot_predictions.extend(scaled_outputs[:, 0])
                overflow_predictions.extend(scaled_outputs[:, 1])

            # Calculate NSE for spigot and overflow
            print("spigot_predictions", len(spigot_predictions))
            print("overflow_predictions", len(overflow_predictions))
            print("spigot_out.shape", spigot_out.shape)
            print("overflow_out.shape", overflow_out.shape)
            spigot_nse, overland_flow_nse = self.compute_nse(spigot_predictions, overflow_predictions, spigot_out, overflow_out)
            mass_in, mass_out = self.compute_mass_balance(ibuc)
            mass_residual = (mass_in - mass_out) / mass_in

            print("Bucket ID:", ibuc)
            print("Spigot NSE:", spigot_nse)
            print("Overflow NSE:", overland_flow_nse)
            print("Mass into the system:", mass_in)
            print("Mass out or left over:", mass_out)
            print(f"Percent mass residual: {mass_residual:.0%}")

            if do_plot_timeseries:
                plot_timeseries(spigot_predictions, overflow_predictions, spigot_out, overflow_out)


