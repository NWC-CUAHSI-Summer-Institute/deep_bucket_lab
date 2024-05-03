import numpy as np
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
#        self.train_data = train_data

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
        predictions_dict = {}
        for i, var in enumerate(self.config['output_vars']):
            predictions_dict[var] = self.scaler_out[var].inverse_transform(output_np[:, i:i+1]).flatten()
        return predictions_dict

    def validate_model(self, do_summary_stats=True, do_individual_bucket_metrics=False, do_plot_timeseries=False):
        output_vars = self.config['output_vars']
        performance_metrics = {var: [] for var in output_vars}
        mass_residuals = []

        for ibuc in self.bucket_dictionary[self.split]['bucket_id'].unique():
            df_obs = self.bucket_dictionary[self.split][self.bucket_dictionary[self.split]['bucket_id'] == ibuc]
            
            # Generate dictionaries for observed outputs right from the DataFrame
            observed = {var: df_obs[var].iloc[self.config['model']['seq_length']-1:-1].to_numpy() for var in output_vars}

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