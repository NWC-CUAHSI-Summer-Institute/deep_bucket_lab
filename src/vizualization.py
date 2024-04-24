#!/home/jonat/anaconda3/envs/deep_bucket_env/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries(spigot_predictions, overflow_predictions, spigot_out, overflow_out):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    # Plot spigot data on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Spigot Output', color=color)
    ax1.plot(spigot_out, color=color, label='Spigot Observation', linestyle='-')
    ax1.plot(spigot_predictions, color=color, label='Spigot Prediction', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.set_xlim([0,100])

    # Create a second y-axis for overflow data
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Overflow Output', color=color)
    ax2.plot(overflow_out, color=color, label='Overflow Observation', linestyle='-')
    ax2.plot(overflow_predictions, color=color, label='Overflow Prediction', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0,100])

    plt.title('Time Series Predictions vs Observations')
    plt.show()
    plt.close()

class Visualization:
    def __init__(self, bucket_dictionary, config, scalers, results=None):
        """
        Initialize the Visualization class.

        Args:
            bucket_dictionary (dict): Dictionary containing the data for each bucket.
            config (dict): Configuration dictionary, possibly containing parameters like seq_length.
            results (dict): Optional. Dictionary containing training results such as losses and RMSEs.
        """
        self.bucket_dictionary = bucket_dictionary
        self.config = config
        self.results = results
        self.input_vars = config.get('input_vars', [])
        self.output_vars = config.get('output_vars', [])
        self.num_epochs = config['model']['num_epochs']
        self.seq_length = config['model']['seq_length']
        self.input_vars = config['input_vars']
        self.output_vars = config['output_vars']
        self.scaler_in, self.scaler_out = scalers

    ######################################################################################
    ######################################################################################
    def viz_simulation(self, split, ibuc, n_plot=100):
        """
        Visualize the inputs, outputs, and predictions for the bucket simulation using a reset index for consistent plotting.

        Args:
            split (str): The data split key ('train', 'val', or 'test').
            ibuc (int): Bucket ID to visualize.
            n_plot (int): Number of points to plot.
        """
        df = self.bucket_dictionary[split][self.bucket_dictionary[split]['bucket_id'] == ibuc]
        df_subset = df.head(n_plot).reset_index(drop=True)  # Reset index and take first n_plot rows

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

        # Plot inputs
        df_subset[self.input_vars].plot(ax=ax1)
        ax1.set_title('Model Inputs')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Input Values')
        ax1.legend(title='Inputs')

        # Plot outputs
        df_subset[self.output_vars].plot(ax=ax2)
        ax2.set_title('Model Outputs')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Output Values')
        ax2.legend(title='Outputs')

        # Optionally, plot h_bucket if it is part of the output_vars and needs a separate axis
        if 'h_bucket' in df.columns:
            ax3 = ax2.twinx()
            df_subset['h_bucket'].plot(ax=ax3, style='g-', label='h_bucket')
            ax3.set_ylabel('Bucket Water Level')
            ax3.legend(title='Water Level', loc='upper left')

        plt.tight_layout()
        plt.show()

    ######################################################################################
    ######################################################################################
    def viz_learning_curve(self):
        """
        Visualize the learning curves for each bucket based on training results.
        """
        if not self.results:
            raise ValueError("Results data is not provided for learning curve visualization.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

        # Loop through results and plot
        for ibuc, metrics in self.results.items():
            ax1.plot(metrics['loss'], label=f'Bucket {ibuc}')
            ax2.plot(metrics['RMSE'], label=f'Bucket {ibuc}')

        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.set_title('RMSE per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    ######################################################################################
    ######################################################################################
    def viz_loader(self, data_loader, n_plot=100):
        inputs_list = []
        outputs_list = []
        
        # Collect all inputs and outputs from the DataLoader
        for inputs, outputs in data_loader:
            inputs_list.append(inputs.detach().cpu().numpy())
            outputs_list.append(outputs.detach().cpu().numpy())

        # Convert lists to numpy arrays
        inputs_array = np.vstack(inputs_list)
        outputs_array = np.vstack(outputs_list)

        # Reshape from 3D [batch_size, seq_length, num_features] to 2D [batch_size * seq_length, num_features]
        inputs_2d = inputs_array.reshape(-1, inputs_array.shape[-1])
        outputs_2d = outputs_array.reshape(-1, outputs_array.shape[-1])

        # Inverse transform to get back to the original scale of data
        inputs_transformed = self.scaler_in.inverse_transform(inputs_2d)
        outputs_transformed = self.scaler_out.inverse_transform(outputs_2d)

        # Plot the last time step of each sequence
        plt.figure(figsize=(6, 4))
        for i in range(inputs_transformed.shape[1]):
            plt.plot(inputs_transformed[-n_plot:, i], label=f'Input Feature {i+1}')
        for i in range(outputs_transformed.shape[1]):
            plt.plot(outputs_transformed[-n_plot:, i], linestyle='--', label=f'Output Feature {i+1}')
        plt.title('Features Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Feature Values')
        plt.legend()
        plt.show()
