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
    
    def scale_model_predictions(self, df_train, output):
        # Rescale predictions to original scale
#       scaled_outputs = self.scaler_out.inverse_transform(output.detach().cpu().numpy())
        output_np = output.detach().cpu().numpy()
        spigot_std = np.std(df_train['q_spigot'])
        spigot_mean = np.mean(df_train['q_spigot'])
        overflow_std = np.std(df_train['q_overflow'])
        overflow_mean = np.mean(df_train['q_overflow'])
        scaled_spigot_predictions = output_np[:, 0] * spigot_std + spigot_mean
        scaled_overflow_predictions = output_np[:, 1] * overflow_std + overflow_mean
        return scaled_spigot_predictions, scaled_overflow_predictions

    def validate_model(self, 
                       do_summary_stats=True, 
                       do_individual_bucket_metrics=False, 
                       do_plot_timeseries=False):
        nse_spigot = []
        nse_overflow = []
        mass_residuals = []
        df_train = self.bucket_dictionary["train"]

        for ibuc in self.bucket_dictionary[self.split]['bucket_id'].unique():
            df_obs = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
            
            # Assume the predictions start after `seq_length` due to the need for initial sequence context
            spigot_out = df_obs['q_spigot'].iloc[self.seq_length:].to_numpy()
            overflow_out = df_obs['q_overflow'].iloc[self.seq_length:].to_numpy()

            spigot_predictions = []
            overflow_predictions = []

            loader = self.loader[ibuc]  # DataLoader for the specific bucket
            for data, _ in loader:
                data = data.to(self.device)
                output = self.model(data)
                
                scaled_spigot, scaled_overflow = self.scale_model_predictions(df_train, output)

                spigot_predictions.extend(scaled_spigot)
                overflow_predictions.extend(scaled_overflow)

            # Calculate NSE for spigot and overflow
            spigot_nse, overflow_nse = self.compute_nse(spigot_predictions, overflow_predictions, spigot_out, overflow_out)
            mass_in, mass_out = self.compute_mass_balance(ibuc)
            mass_residual = (mass_in - mass_out) / mass_in
            nse_spigot.append(spigot_nse)
            nse_overflow.append(overflow_nse)
            mass_residuals.append(mass_residual)

            if do_individual_bucket_metrics:
                print("Bucket ID:", ibuc)
                print("Spigot NSE:", spigot_nse)
                print("Overflow NSE:", overflow_nse)
                # print("Mass into the system:", mass_in)
                # print("Mass out or left over:", mass_out)
                print(f"Percent mass residual: {mass_residual:.0%}")

            if do_plot_timeseries:
                plot_timeseries(spigot_predictions, overflow_predictions, spigot_out, overflow_out)

        if do_summary_stats:
            print("Performance Metrics Summary Across Buckets:")
            print("Spigot NSE - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
                np.mean(nse_spigot), np.median(nse_spigot), np.percentile(nse_spigot, 10), np.percentile(nse_spigot, 90)))
            print("Overflow NSE - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
                np.mean(nse_overflow), np.median(nse_overflow), np.percentile(nse_overflow, 10), np.percentile(nse_overflow, 90)))
            print("Mass Residual - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
                np.mean(mass_residuals), np.median(mass_residuals), np.percentile(mass_residuals, 10), np.percentile(mass_residuals, 90)))

