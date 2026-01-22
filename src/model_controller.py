import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from lstm import deep_bucket_model


class ModelController:
    def __init__(self, config, device, bucket_dictionary):
        self.device = device
        self.config = config
        self.bucket_dictionary = bucket_dictionary

        self.do_config()

        # (Optional) reproducibility for model init
        torch.manual_seed(1)

        self.initialize_model()

        self.scaler_in = None
        self.scaler_out = None
        self.fit_scalerz()

    def initialize_model(self):
        self.lstm = deep_bucket_model(self.model_config).to(self.device)

    def _compute_target_stats(self, train_loader):
        """
        Compute dataset-level mean/std for the *last timestep* target vector
        across all buckets. Targets are in SCALED space (because make_data_loader
        applies scaler_out).
        """
        eps = 1e-8
        ys = []

        for _, loader in train_loader.items():
            for _, targets in loader:
                # targets: (batch, seq_len, out_dim) -> use last step to match training
                ys.append(targets[:, -1, :])

        y = torch.cat(ys, dim=0).to(self.device)  # (N, out_dim)
        y_mean = y.mean(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True) + eps
        return y_mean, y_std

    def do_config(self):
        self.model_config = self.config["model"]
        self.input_vars = self.config["input_vars"]
        self.output_vars = self.config["output_vars"]
        self.seq_length = self.config["model"]["seq_length"]

    def fit_scalerz(self):
        B = self.bucket_dictionary["train"]

        # One scaler for all inputs
        self.scaler_in = StandardScaler()
        self.scaler_in.fit(B[self.input_vars])

        # One scaler per output var
        self.scaler_out = {}
        for var in self.output_vars:
            scaler = StandardScaler()
            scaler.fit(B[[var]])  # keep DataFrame shape
            self.scaler_out[var] = scaler

    def make_data_loader(self, split):
        bucket_list = self.bucket_dictionary[split]["bucket_id"].unique()
        loader = {}

        for ibuc in bucket_list:
            df = self.bucket_dictionary[split][self.bucket_dictionary[split]["bucket_id"] == ibuc]
            if df.empty:
                continue

            # Scale inputs
            data_in = self.scaler_in.transform(df[self.input_vars])

            # Scale outputs per variable
            data_out = np.column_stack([self.scaler_out[var].transform(df[[var]]) for var in self.output_vars])

            seq_length = self.lstm.seq_length
            n = len(data_in)

            # Create sequences
            np_seq_X = np.array([data_in[i : i + seq_length] for i in range(n - seq_length)])
            np_seq_y = np.array([data_out[i : i + seq_length] for i in range(n - seq_length)])

            if np_seq_X.size == 0 or np_seq_y.size == 0:
                continue

            ds = TensorDataset(
                torch.tensor(np_seq_X, dtype=torch.float32),
                torch.tensor(np_seq_y, dtype=torch.float32),
            )
            loader[ibuc] = DataLoader(ds, batch_size=self.lstm.batch_size, shuffle=False)

        return loader

    def train_model(self, train_loader):
        criterion = torch.nn.MSELoss()

        # Compute dataset-level target stats once (across all training buckets), in SCALED space
        y_mean, y_std = self._compute_target_stats(train_loader)

        # Read mass constraint weight from YAML (defaults to 0.0 = off)
        lambda_mass = float(self.config.get("loss", {}).get("lambda_mass", 0.0))

        # Build tensors to inverse-transform outputs back to physical units:
        # StandardScaler: x_scaled = (x - mean) / scale  ->  x = x_scaled * scale + mean
        out_means = torch.tensor(
            [self.scaler_out[v].mean_[0] for v in self.output_vars],
            device=self.device,
            dtype=torch.float32,
        )
        out_scales = torch.tensor(
            [self.scaler_out[v].scale_[0] for v in self.output_vars],
            device=self.device,
            dtype=torch.float32,
        )

        # Indices for mass conservation (only if these vars exist)
        idx_spigot = self.output_vars.index("q_spigot") if "q_spigot" in self.output_vars else None
        idx_overflow = self.output_vars.index("q_overflow") if "q_overflow" in self.output_vars else None
        idx_total = self.output_vars.index("q_total") if "q_total" in self.output_vars else None
        use_mass_loss = (idx_spigot is not None) and (idx_overflow is not None) and (idx_total is not None)

        optimizer = torch.optim.Adam(
            self.lstm.parameters(),
            lr=self.config["model"]["learning_rate"]["start"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["model"]["learning_rate"]["step_size"],
            gamma=self.config["model"]["learning_rate"]["gamma"],
        )

        eps = 1e-8
        num_epochs = self.config["model"]["num_epochs"]

        for epoch in range(num_epochs):
            epoch_losses = []

            for _, loader in train_loader.items():
                bucket_losses = []

                for data, targets in loader:
                    optimizer.zero_grad()

                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    output = self.lstm(data)          # (batch, out_dim) in SCALED space
                    targets_last = targets[:, -1, :]  # (batch, out_dim) in SCALED space

                    # Normalize across outputs using dataset-level stats (SCALED space)
                    targets_n = (targets_last - y_mean) / y_std
                    output_n = (output - y_mean) / y_std

                    data_loss = criterion(output_n, targets_n)

                    # Optional mass penalty in PHYSICAL space
                    if use_mass_loss and lambda_mass > 0.0:
                        # Inverse-transform selected outputs to physical units
                        q_sp = output[:, idx_spigot] * out_scales[idx_spigot] + out_means[idx_spigot]
                        q_ov = output[:, idx_overflow] * out_scales[idx_overflow] + out_means[idx_overflow]
                        q_to = output[:, idx_total] * out_scales[idx_total] + out_means[idx_total]

                        mass_err = q_to - (q_sp + q_ov)

                        # Normalize by physical std of q_total (i.e., its scaler scale) for stability
                        mass_err_n = mass_err / (out_scales[idx_total] + eps)
                        mass_loss = (mass_err_n ** 2).mean()

                        loss = data_loss + lambda_mass * mass_loss
                    else:
                        loss = data_loss

                    loss.backward()
                    optimizer.step()
                    bucket_losses.append(loss.item())

                avg_loss = sum(bucket_losses) / len(bucket_losses)
                epoch_losses.append(avg_loss)

            scheduler.step()
            total_avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}: Total Avg Loss: {total_avg_loss}")

        return self.lstm
