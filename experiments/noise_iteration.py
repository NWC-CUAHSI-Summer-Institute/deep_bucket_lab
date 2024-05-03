import yaml
import subprocess
import os

# This ensures that the directory where this script is located is the base for relative paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels
script_path = os.path.join(base_dir, 'run', 'run_deep_bucket_lab.py')

def update_config_and_run(parameter, values):
    config_path = './configuration/configuration.yml'  # Update this to the path of your config file
    base_noise = 0.01  # Base noise level for all other parameters
    for value in values:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set all noise values to base noise first
        for key in config['synthetic_data']['val']['noise'].keys():
            config['synthetic_data']['val']['noise'][key] = base_noise
        
        # Update the specific noise parameter for validation
        config['synthetic_data']['val']['noise'][parameter] = value
        
        # Print current noise configuration for validation
        print(f"{parameter} noise {value}")
        
        # Save the modified configuration
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        
        # Build the command to run the Python script
        command = ["python3", script_path]

        # Execute the command
        subprocess.run(command, check=True)

# Noise parameters and their range of values
noise_parameters = {
    'pet': [0.2, 0.4],
    'et': [0.2, 0.4],
    'q': [0.2, 0.4],
    'head': [0.2, 0.4]
}

for parameter, values in noise_parameters.items():
    update_config_and_run(parameter, values)
