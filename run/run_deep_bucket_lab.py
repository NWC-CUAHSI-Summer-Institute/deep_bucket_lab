import sys
import os
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch

# ---- Repo-root + src path (no hardcoded cwd assumptions) ----
REPO_ROOT = Path(__file__).resolve().parents[1]          # .../deep_bucket_lab
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))                         # so "from data_generation import ..." works

from data_generation import BucketSimulation
from model_controller import ModelController
from validation import ModelValidator


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path_from_repo(p: str) -> str:
    """Resolve a config path relative to repo root unless it is already absolute."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((REPO_ROOT / pp).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configuration" / "tune_seq48_h64.yml"),
        help="Path to YAML config file (relative to repo root is OK)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Seed first
    set_seed(args.seed)

    # Resolve config path (works from any CWD)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    # Load config
    config = load_config(config_path)

    # Resolve file paths inside config so they work from any CWD
    if "unit_hydrograph_distribution_file" in config:
        config["unit_hydrograph_distribution_file"] = resolve_path_from_repo(
            config["unit_hydrograph_distribution_file"]
        )

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"] else "cpu"
    )

    # Initialize simulations
    bucket_sim_train = BucketSimulation(config, "train")
    bucket_sim_val = BucketSimulation(config, "val")
    bucket_sim_test = BucketSimulation(config, "test")

    # Generate data
    print("generating training data")
    train_data = bucket_sim_train.generate_data(config["synthetic_data"]["train"]["num_records"])

    print("generating val data")
    val_data = bucket_sim_val.generate_data(config["synthetic_data"]["val"]["num_records"])

    print("generating test data")
    test_data = bucket_sim_test.generate_data(config["synthetic_data"]["test"]["num_records"])

    bucket_dictionary = {"train": train_data, "val": val_data, "test": test_data}

    # Train
    model_controller = ModelController(config, device, bucket_dictionary)

    train_loader = model_controller.make_data_loader("train")
    val_loader = model_controller.make_data_loader("val")
    test_loader = model_controller.make_data_loader("test")

    print("training model")
    trained_model = model_controller.train_model(train_loader)

    # Validate
    model_validator = ModelValidator(
        trained_model,
        device,
        bucket_dictionary,
        val_loader,
        config,
        "val",
        model_controller.scaler_out,
    )

    print("validating model")
    model_validator.validate_model()
