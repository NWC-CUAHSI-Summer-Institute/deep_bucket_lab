#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import torch
import numpy as np
import yaml

class ModelValidator:
    def __init__(self, lstm, device, bucket_dictionary, config, split):
        """
        Initialize the ModelValidator with split-specific configuration.

        Args:
            lstm (torch.nn.Module): The LSTM model to be validated.
            device (torch.device): The device (GPU/CPU) the model is running on.
            bucket_dictionary (dict): Has the stuff
            config (dict): Configuration dictionary loaded from the YAML file.
            split (str): 'train', 'val', or 'test' to indicate the data split.
        """
        self.model = lstm
        self.device = device
        self.config = config['synthetic_data'][split]
        self.bucket_dictionary = bucket_dictionary
        self.train_start = self.config['num_records']  # Adjust based on actual config structure
        self.train_end = self.config['num_records']  # Adjust as needed
        self.seq_length = config['model']['seq_length']

    def make_prediction(self, np_val_seq_X, ibuc):
        model_output_val = self.model(torch.Tensor(np_val_seq_X[ibuc]).to(self.device))
        val_spigot_prediction = []
        val_overflow_prediction = []
        df = self.bucket_dictionary[ibuc]
        for i in range(model_output_val.shape[0]):
            val_spigot_prediction.append((model_output_val[i, -1, 1].cpu().detach().numpy() * \
                                          np.std(df.loc[self.train_start:self.train_end, 'q_spigot'])) + \
                                         np.mean(df.loc[self.train_start:self.train_end, 'q_spigot']))
            val_overflow_prediction.append((model_output_val[i, -1, 0].cpu().detach().numpy() * \
                                            np.std(df.loc[self.train_start:self.train_end, 'q_overflow'])) + \
                                           np.mean(df.loc[self.train_start:self.train_end, 'q_overflow']))
        return val_spigot_prediction, val_overflow_prediction


    def compute_nse(self, val_spigot_prediction, val_overflow_prediction, ibuc):
        df = self.bucket_dictionary[ibuc]
        seq_length = self.model.seq_length  # Ensure you have access to seq_length

        # Adjust actual outputs to start from seq_length to match the predictions
        spigot_out = df['q_spigot'].iloc[seq_length:].to_numpy()
        overflow_out = df['q_overflow'].iloc[seq_length:].to_numpy()

        # Check lengths match
        if len(val_spigot_prediction) != len(spigot_out):
            raise ValueError(f"Length mismatch between predicted spigot output ({len(val_spigot_prediction)}) and actual spigot output ({len(spigot_out)})")

        if len(val_overflow_prediction) != len(overflow_out):
            raise ValueError(f"Length mismatch between predicted overflow output ({len(val_overflow_prediction)}) and actual overflow output ({len(overflow_out)})")

        # Calculate variances for NSE
        spigot_pred_variance = np.sum(np.power(np.array(val_spigot_prediction) - spigot_out, 2))
        spigot_obs_variance = np.sum(np.power(spigot_out.mean() - spigot_out, 2))
        overflow_pred_variance = np.sum(np.power(np.array(val_overflow_prediction) - overflow_out, 2))
        overflow_obs_variance = np.sum(np.power(overflow_out.mean() - overflow_out, 2))

        # Calculate NSE
        spigot_nse = 1 - (spigot_pred_variance / spigot_obs_variance) if spigot_obs_variance != 0 else float('nan')
        overland_flow_nse = 1 - (overflow_pred_variance / overflow_obs_variance) if overflow_obs_variance != 0 else float('nan')

        return spigot_nse, overland_flow_nse

    def compute_mass_balance(self, ibuc):
        df = self.bucket_dictionary[ibuc]
        mass_in = df['precip'].sum()
        mass_out = df['et'].sum() + df['q_overflow'].sum() + df['q_spigot'].sum() + df.loc[df.index[-1], 'h_bucket']
        return mass_in, mass_out

    def post_process_predictions(self, model_outputs):
        # Implement any necessary transformation or extraction of predictions from model outputs
        # This is a placeholder function and needs actual implementation based on model output structure
        spigot_predictions = [output[0] for output in model_outputs]
        overflow_predictions = [output[1] for output in model_outputs]
        return spigot_predictions, overflow_predictions

    def validate_model(self, data_loader, ibuc):
        val_spigot_predictions = []
        val_overflow_predictions = []

        for data in data_loader:
            inputs, _ = data
            outputs = self.model(inputs.to(self.device))
            spigot_predictions, overflow_predictions = outputs[:, 0], outputs[:, 1]

            # Detach from the computation graph and convert to NumPy right here if not done earlier
            if spigot_predictions.requires_grad:
                spigot_predictions = spigot_predictions.detach().cpu().numpy()
            if overflow_predictions.requires_grad:
                overflow_predictions = overflow_predictions.detach().cpu().numpy()

            val_spigot_predictions.extend(spigot_predictions)
            val_overflow_predictions.extend(overflow_predictions)

        spigot_nse, overland_flow_nse = self.compute_nse(val_spigot_predictions, val_overflow_predictions, ibuc)
        mass_in, mass_out = self.compute_mass_balance(ibuc)
        mass_residual = (mass_in - mass_out) / mass_in

        print("Spigot NSE:", spigot_nse)
        print("Overflow NSE:", overland_flow_nse)
        print("Mass into the system:", mass_in)
        print("Mass out or left over:", mass_out)
        print(f"Percent mass residual: {mass_residual:.0%}")



# Example usage assuming configurations and model are set up
