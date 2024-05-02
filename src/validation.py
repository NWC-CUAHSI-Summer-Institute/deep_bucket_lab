#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import numpy as np
import yaml
from vizualization import plot_timeseries

class ModelValidator:
    def __init__(self, lstm, 
                 device, 
                 bucket_dictionary, 
                 loader, 
                 config, 
                 split, 
                 scaler_out):
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
        self.config = config
        self.bucket_dictionary = bucket_dictionary
        self.loader = loader
        self.split = split
        self.scaler_out = scaler_out

    def compute_nse(self, prediction, observed):
        # Compute NSE for a single output variable
        pred_variance = np.sum(np.power(np.array(prediction) - observed, 2))
        obs_variance = np.sum(np.power(observed.mean() - observed, 2))
        return 1 - (pred_variance / obs_variance) if obs_variance != 0 else float('nan')

    def compute_mass_balance(self, ibuc):
        df = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
        mass_in = df['precip'].sum()
        mass_out = df['et'].sum() + df['q_overflow'].sum() + df['q_spigot'].sum() + df.loc[df.index[-1], 'h_bucket']
        return mass_in, mass_out
    
    def scale_model_predictions(self, output):
        output_np = output.detach().cpu().numpy()
        scaled_outputs = self.scaler_out.inverse_transform(output_np)
        predictions_dict = {}
        for i, var in enumerate(self.config['output_vars']):
            predictions_dict[var] = scaled_outputs[:, i]  # Ensure ordering matches config['output_vars']
        return predictions_dict

    def validate_model(self, do_summary_stats=True, do_individual_bucket_metrics=False, do_plot_timeseries=False):
        output_vars = self.config['output_vars']
        performance_metrics = {var: [] for var in output_vars}
        mass_residuals = []

        for ibuc in self.bucket_dictionary[self.split]['bucket_id'].unique():
            df_obs = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
            
            # Generate dictionaries for observed outputs right from the DataFrame
            observed = {var: df_obs[var].iloc[self.config['model']['seq_length']:].to_numpy() for var in output_vars}

            # Initialize dictionaries to collect predictions for each output variable
            predictions = {var: [] for var in output_vars}

            loader = self.loader[ibuc]
            for data, _ in loader:
                data = data.to(self.device)
                output = self.model(data)
                scaled_output = self.scale_model_predictions(output)  # Ensure this method appropriately handles scaling for all variables

                for i, var in enumerate(output_vars):
                    predictions[var].extend(scaled_output[var])

            # Calculate NSE for each variable
            nse_values = {}
            for var in output_vars:
                nse = self.compute_nse(predictions[var], observed[var])
                performance_metrics[var].append(nse)
                nse_values[var] = nse

            mass_in, mass_out = self.compute_mass_balance(ibuc)
            mass_residual = (mass_in - mass_out) / mass_in
            mass_residuals.append(mass_residual)

            if do_individual_bucket_metrics:
                print(f"Bucket ID: {ibuc}")
                for var in output_vars:
                    print(f"{var} NSE:", nse_values[var])
                print(f"Percent mass residual: {mass_residual:.0%}")

            if do_plot_timeseries:
                plot_timeseries(predictions, observed, output_vars)

        if do_summary_stats:
            print("Performance Metrics Summary Across Buckets:")
            for var in output_vars:
                print(f"{var} NSE - Mean: {np.mean(performance_metrics[var]):.3f}, "
                    f"Median: {np.median(performance_metrics[var]):.3f}, "
                    f"10th Pctl: {np.percentile(performance_metrics[var], 10):.3f}, "
                    f"90th Pctl: {np.percentile(performance_metrics[var], 90):.3f}")
            print("Mass Residual - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
                np.mean(mass_residuals), np.median(mass_residuals), np.percentile(mass_residuals, 10), np.percentile(mass_residuals, 90)))

    # def validate_model(self, 
    #                    do_summary_stats=True, 
    #                    do_individual_bucket_metrics=False, 
    #                    do_plot_timeseries=False):
    #     nse_spigot = []
    #     nse_overflow = []
    #     mass_residuals = []
    #     df_train = self.bucket_dictionary["train"]

    #     for ibuc in self.bucket_dictionary[self.split]['bucket_id'].unique():
    #         df_obs = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
            
    #         # Assume the predictions start after `seq_length` due to the need for initial sequence context
    #         spigot_out = df_obs['q_spigot'].iloc[self.seq_length:].to_numpy()
    #         overflow_out = df_obs['q_overflow'].iloc[self.seq_length:].to_numpy()

    #         spigot_predictions = []
    #         overflow_predictions = []

    #         loader = self.loader[ibuc]  # DataLoader for the specific bucket
    #         for data, _ in loader:
    #             data = data.to(self.device)
    #             output = self.model(data)
                
    #             scaled_spigot, scaled_overflow = self.scale_model_predictions(output)

    #             spigot_predictions.extend(scaled_spigot)
    #             overflow_predictions.extend(scaled_overflow)

    #         # Calculate NSE for spigot and overflow
    #         spigot_nse, overflow_nse = self.compute_nse(spigot_predictions, overflow_predictions, spigot_out, overflow_out)
    #         mass_in, mass_out = self.compute_mass_balance(ibuc)
    #         mass_residual = (mass_in - mass_out) / mass_in
    #         nse_spigot.append(spigot_nse)
    #         nse_overflow.append(overflow_nse)
    #         mass_residuals.append(mass_residual)

    #         if do_individual_bucket_metrics:
    #             print("Bucket ID:", ibuc)
    #             print("Spigot NSE:", spigot_nse)
    #             print("Overflow NSE:", overflow_nse)
    #             # print("Mass into the system:", mass_in)
    #             # print("Mass out or left over:", mass_out)
    #             print(f"Percent mass residual: {mass_residual:.0%}")

    #         if do_plot_timeseries:
    #             plot_timeseries(spigot_predictions, overflow_predictions, spigot_out, overflow_out)

    #     if do_summary_stats:
    #         print("Performance Metrics Summary Across Buckets:")
    #         print("Spigot NSE - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
    #             np.mean(nse_spigot), np.median(nse_spigot), np.percentile(nse_spigot, 10), np.percentile(nse_spigot, 90)))
    #         print("Overflow NSE - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
    #             np.mean(nse_overflow), np.median(nse_overflow), np.percentile(nse_overflow, 10), np.percentile(nse_overflow, 90)))
    #         print("Mass Residual - Mean: {:.3f}, Median: {:.3f}, 10th Pctl: {:.3f}, 90th Pctl: {:.3f}".format(
    #             np.mean(mass_residuals), np.median(mass_residuals), np.percentile(mass_residuals, 10), np.percentile(mass_residuals, 90)))

