import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 
class ANN_Model(nn.Module):
    """
    A customizable Artificial Neural Network (ANN) model with configurable activation functions, 
    batch normalization, and multiple hidden layers.

    Parameters:
    - input_size: int, the size of the input layer.
    - hidden_size: int, the size of the hidden layers.
    - output_size: int, the size of the output layer.
    - args: Namespace, configuration arguments including:
        - ann_num_layers: int, number of hidden layers.
        - ann_act: str, activation function ('relu', 'tanh', 'sigmoid', 'linear').
        - ann_batch_norm: bool, whether to use batch normalization.

    Methods:
    - forward(x): Performs the forward pass of the ANN.
    """
    def __init__(self, input_size, hidden_size, output_size, args):
        super(ANN_Model, self).__init__()
        self.args = args
        # Store the layers
        layers = []
        batch_norms = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        batch_norms.append(nn.BatchNorm1d(hidden_size))

        # Hidden layers with batch normalization
        for i in range(args.ann_num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            batch_norms.append(nn.BatchNorm1d(hidden_size))

        # Output layer (no batch normalization for the output)
        layers.append(nn.Linear(hidden_size, output_size))

        # Store layers and batch normalization layers in ModuleList
        self.layers = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_norms)

        # Define the activation function based on args
        if args.ann_act == 'relu':
            self.activation = nn.ReLU()
        elif args.ann_act == 'tanh':
            self.activation = torch.tanh
        elif args.ann_act == 'sigmoid':
            self.activation = torch.sigmoid
        elif args.ann_act == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError("Unsupported activation function")

        # define the output activation
        self.out_activation = lambda x: x

    def forward(self, x):
        # Apply layers with batch normalization and activations
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.args.ann_batch_norm:
                x = self.batch_norms[i](x)  # Apply batch normalization
            x = self.activation(x)      # Apply activation function

        # Apply the last layer (output layer) without batch normalization
        x = self.layers[-1](x)
        x = self.out_activation(x)     # Apply output activation
        return x

class EarlyStopping:
    """
    Monitors validation loss during training and stops training early if no improvement 
    is observed after a specified number of epochs (patience).

    Parameters:
    - patience: int, the number of epochs to wait for improvement before stopping.
    - verbose: bool, if True, prints messages when validation loss improves.
    - delta: float, minimum change in validation loss to be considered an improvement.
    - path: str, file path to save the best model checkpoint.
    - trace_func: function, function to output trace messages (e.g., print).

    Methods:
    - __call__(val_loss, model): Checks validation loss and triggers early stopping or saves a checkpoint.
    - save_checkpoint(val_loss, model): Saves the model if validation loss improves.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def train_model(model, train_loader, val_loader, loss_func, optimizer, epochs, args, scheduler=None, checkpoint_dir='./'):
    """
    Trains the ANN model with optional early stopping and checkpointing.

    Parameters:
    - model: nn.Module, the ANN model to be trained.
    - train_loader: DataLoader, DataLoader for the training dataset.
    - val_loader: DataLoader, DataLoader for the validation dataset.
    - loss_func: nn.Module, loss function for training (e.g., MSELoss).
    - optimizer: torch.optim.Optimizer, optimizer for the model (e.g., Adam).
    - epochs: int, number of epochs for training.
    - args: Namespace, configuration arguments including:
        - ann_early_stop: bool, whether to enable early stopping.
        - ann_patience: int, patience for early stopping.
        - ann_min_delta: float, minimum improvement for early stopping.
    - scheduler: torch.optim.lr_scheduler, learning rate scheduler (optional).
    - checkpoint_dir: str, directory to save model checkpoints.

    Returns:
    - model: nn.Module, the trained model loaded with the best weights (if early stopping is used).
    """
    # 
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pt')
    # 
    if args.ann_early_stop:
        early_stopping = EarlyStopping(patience=args.ann_patience, delta=args.ann_min_delta, path=checkpoint_path)

    for epoch in range(epochs):
        # training
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = loss_func(outputs, batch_Y)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            total_train_loss += loss.item()
        #
        train_loss = total_train_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        # 
        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                val_loss = loss_func(val_outputs, Y_val_batch)
                total_val_loss += val_loss.item()
        #
        val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Elapsed Time: {elapsed_time}')

        # Check for early stopping
        early_stopping(val_loss, model)

        # if we want to early stop
        if early_stopping.early_stop and args.ann_early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # load the last checkpoint (with the best model if early stop)
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Saves a checkpoint of the model and optimizer state.

    Parameters:
    - model: nn.Module, the model to save.
    - optimizer: torch.optim.Optimizer, the optimizer to save.
    - epoch: int, the current epoch.
    - loss: float, the loss value at the time of saving.
    - checkpoint_path: str, the file path to save the checkpoint.

    Saves:
    - A dictionary containing the model state, optimizer state, epoch, and loss.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


