import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class FullyConnectedPredictor(nn.Module):
    def __init__(self, config, n_in=15, n_out=42, dropout_prob=0.3):
        """
        A Fully Connected MLP for predicting electron density profiles
        Args:
            config: List defining the number of neurons in each hidden layer
            n_in: Number of input features
            n_out: Number of output features
            dropout_prob: Dropout probability for regularization
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_prob = dropout_prob

        # Define the network structure
        self.config = config

        # Build the modules
        self.classifier = self._make_sequential(self.n_in, self.config, self.n_out)
        
        # Initialize weights
        self._initialize_weights()

    def _make_sequential(self, in_dim, config, out_dim):
        """
        Helper function to build a sequence of layers
        """
        layers = []
        current_dim = in_dim
        
        for layer_dim in config:
            layers.extend([
                nn.Linear(current_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob)
            ])
            current_dim = layer_dim
            
        # The final layer
        layers.append(nn.Linear(current_dim, out_dim))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass: Encode input conditions, then decode into a full profile
        """
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        Initializes weights using He (Kaiming) initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Weight initialization finished!')


    def evaluate(self, loader, loss_fn, device):
        """
        Evaluation method for datasets with true labels
        Input:
            loader: DataLoader providing (features, true, stds, height_indices)
            loss_fn: Loss function to compute the loss
            device: Device to run the computations on (CPU or GPU)
        Output:
            avg_loss: Average loss over the dataset
        """
        self.eval()
        total_loss = 0.
        self.true, self.preds, self.height_indices = [], [], []
        with torch.no_grad():
            for features, true, stds, height_indices in loader:
                features = features.to(device)
                true = true.to(device)
                stds = stds.to(device)
                height_indices = height_indices.to(device).long()

                preds = self(features)
                loss = loss_fn(preds, true, stds, height_indices)
                total_loss += loss.item()

                self.true.extend(true.detach().cpu().tolist())
                self.preds.extend(preds.detach().cpu().tolist())
                self.height_indices.extend(height_indices.detach().cpu().tolist())

        return total_loss / len(loader)
    
    def evaluate_case(self, loader, device):
        """
        Evaluation method for VHF cases without true labels
        Input:
            loader: DataLoader providing (features, stds, height_indices)
            device: Device to run the computations on (CPU or GPU)  
        Output:
            preds: List of predicted profiles
            height_indices: List of height indices corresponding to input Ne
        """
        self.eval()
        self.preds, self.height_indices = [], []
        with torch.no_grad():
            for features, stds, height_indices in loader:
                features = features.to(device)
                stds = stds.to(device)
                height_indices = height_indices.to(device).long()

                preds = self(features)

                self.preds.extend(preds.detach().cpu().tolist())
                self.height_indices.extend(height_indices.detach().cpu().tolist())
        return self.preds, self.height_indices
    
class InMemorySingleFileDataset(Dataset):
    def __init__(self, file_path, split='train'):
        """
        A PyTorch dataset that loads a specific data split from a single 
        .npz file into RAM.

        Args:
            file_path (str): Path to the single .npz data file.
            split (str): The data split to load. Should be 'train', 'validation', or 'test'.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        
        with np.load(file_path) as data:
            features_key = f'{split}_features'
            profiles_key = f'{split}_labels'
            std_devs_key = f'{split}_ne_std'
            
            all_features = data[features_key][:, :-1].astype(np.float32)
            all_profiles = data[profiles_key].astype(np.float32)
            all_std_devs = data[std_devs_key].astype(np.float32)
            all_height_indices = data[features_key][:, -1:]


        # Convert to PyTorch Tensors and store 
        self.features = torch.from_numpy(all_features)
        self.profiles = torch.from_numpy(all_profiles)
        self.std_devs = torch.from_numpy(all_std_devs)
        self.height_indices = torch.from_numpy(all_height_indices)
        
        print(f"'{split}' split loading complete. Loaded {len(self)} samples.")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.profiles[idx],
            self.std_devs[idx],
            self.height_indices[idx]
        )
    

class InMemoryCaseDataset(Dataset):
    def __init__(self, file_path):
        """
        A PyTorch dataset that loads a specific .npz file into RAM.
        For use with VHF cases without true labels.

        Args:
            file_path (str): Path to the single .npz data file.
        """
        
        with np.load(file_path) as data:
            features_key = f'features'
            std_devs_key = f'stds'
            
            all_features = data[features_key][:, :-1].astype(np.float32)
            all_std_devs = data[std_devs_key].astype(np.float32)
            all_height_indices = data[features_key][:, -1:]

        self.features = torch.from_numpy(all_features)
        self.std_devs = torch.from_numpy(all_std_devs)
        self.height_indices = torch.from_numpy(all_height_indices)
        
        print(f"Data loading complete. Loaded {len(self)} samples.")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.std_devs[idx],
            self.height_indices[idx]
        )
    
      
class WeightedMSELoss(torch.nn.Module):
    def __init__(self, additional_weight, epsilon=1e-8, min_std=1e-3):
        """
        Custom weighted MSE loss.
        Args: 
            additional_weight (float): Additional weight to apply to the measurement point error.
            epsilon (float): a small value to add to the denominator to prevent division by zero.
            min_std (float): a small value to clamp the SD to prevent NaN values.
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_std = min_std
        self.additional_weight = additional_weight

    def forward(self, y_pred, y_true, stds, meas_idx):
        """
        Args:
            y_pred (torch.Tensor): The model's predictions. Shape: (batch_size, points)
            y_true (torch.Tensor): The ground truth labels. Shape: (batch_size, points)
            stds (torch.Tensor): The standard deviations for the labels. Shape: (batch_size, points)
            meas_idx (torch.Tensor): A tensor of indices for the measurement point for each sample.
                                     Shape: (batch_size,)
        """
        # Clamp stds to prevent division by very small numbers
        stds = torch.clamp(stds, min=self.min_std)

        # Calculate the base weighted squared error for all points
        squared_error = (y_pred - y_true) ** 2
        inv_variance = 1.0 / (stds.pow(2) + self.epsilon)
        weighted_squared_error = inv_variance * squared_error

        # Calculate the standard loss (mean over all points)
        loss = torch.mean(weighted_squared_error)

        if self.additional_weight > 0:
            # Create a batch index tensor on the same device as the inputs
            batch_indices = torch.arange(y_true.size(0), device=y_true.device)

            # Efficiently gather the specific weighted errors for the measurement points
            meas_idx = meas_idx.long()
            meas_errors = weighted_squared_error[batch_indices, meas_idx]

            # Calculate the additional penalty for the measurement points
            additional_loss = torch.mean(meas_errors * self.additional_weight)
            loss += additional_loss

        # Check for NaN loss
        if torch.isnan(loss):
            raise ValueError("Loss is NaN! Check your input data for NaNs or Infs.")

        return loss
    
