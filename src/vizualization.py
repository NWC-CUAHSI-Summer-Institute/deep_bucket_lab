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

    # Create a second y-axis for overflow data
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Overflow Output', color=color)
    ax2.plot(overflow_out, color=color, label='Overflow Observation', linestyle='-')
    ax2.plot(overflow_predictions, color=color, label='Overflow Prediction', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Time Series Predictions vs Observations')
    plt.show()
    plt.close()

class Visualization:
    def __init__(self, bucket_dictionary, config, results=None):
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
        """
        Visualize the last timestep inputs and outputs from the DataLoader.

        Args:
            data_loader (DataLoader): The DataLoader containing the data to visualize.
            n_plot (int): Number of data points to plot.
        """
        # Concatenate all batches to recreate the sequence
        inputs_list = []
        outputs_list = []
        for inputs, outputs in data_loader:
            inputs_list.append(inputs.numpy())
            outputs_list.append(outputs.numpy())

        # Convert list to arrays
        inputs_array = np.vstack(inputs_list)
        outputs_array = np.vstack(outputs_list)

        # Extract only the last timestep of each sequence
        # Assuming that each batch is (batch_size, seq_length, num_features)
        last_inputs = inputs_array[:, -1, :]  # Take the last timestep from each sequence
        last_outputs = outputs_array  # Assuming outputs already correspond to the last timestep

        # Ensure we don't exceed the number of available points
        n_plot = min(n_plot, last_inputs.shape[0])

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

        # Plot last timestep of inputs
        for i in range(last_inputs.shape[1]):
            ax1.plot(range(n_plot), last_inputs[:n_plot, i], label=self.input_vars[i] if i < len(self.input_vars) else f"Feature {i}")
        ax1.set_title('Last Timestep Inputs over Time')
        ax1.legend()
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Input Values at Last Timestep')

        # Plot outputs
        for i in range(last_outputs.shape[1]):
            ax2.plot(range(n_plot), last_outputs[:n_plot, i], label=self.output_vars[i] if i < len(self.output_vars) else f"Feature {i}")
        ax2.set_title('Outputs over Time')
        ax2.legend()
        ax2.set_xlabel('Time steps')
        ax2.set_ylabel('Output Values')

        plt.tight_layout()
        plt.show()
        plt.close()
    
