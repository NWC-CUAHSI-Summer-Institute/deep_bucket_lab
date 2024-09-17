import sys
sys.path.append('./src/')
import yaml
import torch

#print(sys.path)
from data_generation import BucketSimulation
from model_controller import ModelController
from validation import ModelValidator

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('./configuration/configuration.yml')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')

# Initialize and generate synthetic data for each split
bucket_sim_train = BucketSimulation(config, 'train')
bucket_sim_val = BucketSimulation(config, 'val')
bucket_sim_test = BucketSimulation(config, 'test')

# Simulate and store data for training, validation, and testing
train_data = bucket_sim_train.generate_data(config['synthetic_data']['train']['num_records'])
val_data = bucket_sim_val.generate_data(config['synthetic_data']['val']['num_records'])
test_data = bucket_sim_test.generate_data(config['synthetic_data']['test']['num_records'])

bucket_dictionary = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}

# Initialize the LSTM model and Model Controller
model_controller = ModelController(config, device, bucket_dictionary)

# Prepare data loaders
train_loader = model_controller.make_data_loader('train')
val_loader = model_controller.make_data_loader('val')
test_loader = model_controller.make_data_loader('test')

# Now train_loader, val_loader, and test_loader should be dictionaries
trained_model = model_controller.train_model(train_loader)

model_validator = ModelValidator(trained_model, device, 
                                 bucket_dictionary, val_loader, 
                                 config, "val", model_controller.scaler_out)
model_validator.validate_model()
