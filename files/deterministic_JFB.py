import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from statistics import mean
import pickle
import os
import random
import re
import math


# Configuration for data and model parameters
class Config:
    def __init__(self, batch_size=None):
        self.dataset = None
        self.inputs = None
        self.outputs = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_size = 3000
        self.train_size = int(0.8 * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size

        # WIDTH OF MODEL
        self.input_dim = 3
        self.latent_dim = 3
        self.output_dim = 1
        self.batch_size = batch_size if batch_size is not None else self.train_size  # Default: full-batch for dataset_size=5000 is 4000
        self.create_dataset()

    def create_dataset(self):
        # Room temperature model at different points in space, domain is dims of the room (in ft)
        x = torch.linspace(0, 20, self.dataset_size) # Length of room
        y = torch.linspace(0, 13, self.dataset_size) # Width of room
        z = torch.linspace(0, 8, self.dataset_size) # Height of room
        self.inputs = torch.stack([x, y, z], dim=1)

        # Model output with noise
        epsilon = 0 # Adjustable error in measurement
        x_err = x + (epsilon * torch.randn_like(x))
        y_err = y + (epsilon * torch.randn_like(y))
        z_err = z + (epsilon * torch.randn_like(z))

        # Target function: heat transfer model with x,y,z representing parameters like net heat input, net heat loss, external environmental factors
        outputs = torch.tanh(x_err / (1 + torch.exp(-y_err))) + y_err / (1 + x_err ** 2 + z_err ** 2)
        self.outputs = outputs.unsqueeze(1)

        self.dataset = TensorDataset(self.inputs, self.outputs)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size], generator=torch.Generator().manual_seed(42))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_size, shuffle=False)

    def update_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


# Implicit network model
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        # Depth of model:
        self.fc = nn.Linear(config.latent_dim, config.latent_dim)  # Take this to be latent layer
        self.input_injection = nn.Linear(config.input_dim, config.latent_dim)
        self.output_layer = nn.Linear(config.latent_dim, config.output_dim)
        self.activation = nn.ReLU()

        self.latent_dim = config.latent_dim

    def transformation(self, u, d):
        return self.activation(self.fc(u) + self.input_injection(d))

    def forward(self, d):
        """
        perform fixed point iteration loop;
        stopping criterion: if the difference (maximum absolute value) between successive outputs is "close enough," we consider this to be convergence;
        for JFB, don't need to track all gradients (only the final iteration's grads);
        after convergence, compute one final time to store final iteration's gradients
        """
        max_iters = 1000
        tol = 1e-3
        u = torch.zeros(d.size(0), self.latent_dim, device=d.device)
        with torch.no_grad():
            for _ in range(max_iters):
                u_next = self.transformation(u, d)
                if torch.linalg.vector_norm(u - u_next, ord=float('inf')) < tol: # Stopping criterion
                    break  # It keeps the output just BEFORE the tolerance is reached
                u = u_next.clone()  # Else, update u's value to be its next iteration's value, and then keep iterating
        u = self.transformation(u, d)
        out = self.output_layer(u)  # Pass through output layer

        return out


class Helper:
    @staticmethod
    def set_seed(seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # If using CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def extract_number(s):
        """
        extracts the first number from a string s
        """
        match = re.search(r'\d+', s)
        return int(match.group()) if match else None

    @staticmethod
    def calculate_max_iters(dataset_size, batch_size):
        return math.ceil(dataset_size / batch_size)


# Class handling training, validation, and plotting
class Trainer:
    def __init__(self, network, configuration, crit, opti):
        self.net = network
        self.config = configuration
        self.criterion = crit
        self.optimizer = opti
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def train_model(self, epochs):
        """
        returns only training dataset statistics stored in lists: losses, accuracies
        but also updates lists storing validation dataset statistics
        """
        # Reinitialize metric lists at the beginning of each training session
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.epoch_times = []
        self.all_batch_total_train_time = []

        self.training_metrics = {
            'training loss': self.training_losses,
            'validation loss': self.validation_losses,
            'training accuracy': self.training_accuracies,
            'validation accuracy': self.validation_accuracies,
            'training time': self.epoch_times
        }

        lowest_loss = float('inf')  # Incorporate storage of lowest loss value
        start_time = time.time()  # Overall training start time

        for epoch in range(epochs):
            epoch_start_time = time.time()  # Start time of current epoch
            accumulated_training_loss = 0.0  # Reset for each epoch
            training_dataset_seen = 0  # Reset for each epoch
            epoch_accuracies = []  # Reset for each epoch
            threshold = 0.1
            self.net.train()

            # First go through each of the batches:
            for x_batch, y_true_batch in self.config.train_dataloader:
                x_batch, y_true_batch = x_batch.to(self.device), y_true_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.net(x_batch)
                loss = self.criterion(y_pred, y_true_batch)
                loss.backward()
                self.optimizer.step()

                # Stats for Loss: Calculate an individual batch's loss weighted by its batch size
                accumulated_training_loss += loss.item() * x_batch.size(0)
                training_dataset_seen += x_batch.size(0)

                # Stats for Accuracy: Calculate an individual batch's mean accuracy measure
                abs_error = torch.abs(y_pred - y_true_batch)
                accuracy = (abs_error <= threshold).float().mean().item()
                epoch_accuracies.append(accuracy) # Adds a proportional measure of how accurate the current batch was to its matched true values

            # Full Epoch Average Training Loss
            average_training_loss = accumulated_training_loss / training_dataset_seen
            self.training_metrics['training loss'].append(average_training_loss)

            # Full Epoch Average Training Accuracy
            self.training_metrics['training accuracy'].append(mean(epoch_accuracies)) # Takes the mean of the list, resulting in a proportion!

            # Update the lowest loss
            if average_training_loss < lowest_loss:
                lowest_loss = average_training_loss
                self.saved_epoch = epoch + 1  # Save the initial epoch number that reached this lowest state
                model_path = f'best_model_state_{self.config.batch_size}.pth'  # Save the model state if this is a new minimum
                torch.save(self.net.state_dict(), model_path)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.epoch_times.append(epoch_duration)

            # Keep track of validation metrics after each epoch: Add each epoch's evaluation to a list for plotting purposes:
            val_loss, val_accuracy = self.validate()
            self.training_metrics['validation loss'].append(val_loss)
            self.training_metrics['validation accuracy'].append(val_accuracy)

            # scheduler.step()  # Update the learning rate
            # scheduler.step(val_loss)  # Update the learning rate

            # Print Statistics
            print(f'Epoch {epoch + 1}: Training Loss = {self.training_metrics["training loss"][-1]:.4f}, Training Accuracy = {self.training_metrics["training accuracy"][-1] * 100:.2f}%, Validation Loss = {self.training_metrics["validation loss"][-1]:.4f}, Validation Accuracy = {self.training_metrics["validation accuracy"][-1] * 100:.2f}%')

        # Overall training time across all epochs
        total_training_time = time.time() - start_time
        self.all_batch_total_train_time.append(total_training_time)
        self.training_time_minutes = total_training_time / 60

        print(f"Finished training dataset with batch size {self.config.batch_size}. Training completed in {self.training_time_minutes:.2f} minutes")
        with open('output.txt', 'a') as file:
            print(f"Lowest loss value is {lowest_loss:.4f} first reached at Epoch {self.saved_epoch}", file=file)

        self.save_training_metrics(self.config.batch_size)

        return self.training_metrics['training loss'], self.training_metrics['training accuracy']

    def validate(self):
        """
        returns individual batch validation statistics which will be added to list during training loop: loss, accuracy
        """
        threshold = 0.1

        # Batch Validation Loop: used for evaluation
        self.net.eval()
        accumulated_validation_loss = 0.0
        val_dataset_seen = 0
        val_accuracies = [] # For keeping track of individual batch accuracy

        with torch.no_grad():
            for x_val_batch, y_val_true in self.config.val_dataloader:
                x_val_batch, y_val_true = x_val_batch.to(self.device), y_val_true.to(self.device)
                y_val_pred = self.net(x_val_batch)

                # Individual Validation Batch Loss
                val_loss = self.criterion(y_val_pred, y_val_true)
                accumulated_validation_loss += val_loss.item() * x_val_batch.size(0)
                val_dataset_seen += x_val_batch.size(0)

                # Individual Validation Batch Accuracy
                val_abs_error = torch.abs(y_val_pred - y_val_true)
                val_accuracy = (val_abs_error <= threshold).float().mean().item() # True = 1.0, False = 0.0: finds proportion of accuracy for a batch and then returns accuracy as a percentage
                val_accuracies.append(val_accuracy)

        # Full Batch Validation Loss
        validation_loss = accumulated_validation_loss / val_dataset_seen

        # Full Batch Validation Accuracy
        average_val_accuracy = (sum(val_accuracies) / len(val_accuracies))
        return validation_loss, average_val_accuracy

    def save_training_metrics(self, batch_size):
        """
        saves statistics file to directory
        """
        filename = f'metrics_batch_{batch_size}.pkl'
        self.training_metrics = {
            'training loss': self.training_losses,
            'validation loss': self.validation_losses,
            'training accuracy': self.training_accuracies,
            'validation accuracy': self.validation_accuracies,
            'training time': self.epoch_times
        }
        with open(filename, 'wb') as f:
            pickle.dump(self.training_metrics, f)

    @staticmethod
    def plot_metrics():
        plt.figure(figsize=(12+5, 5+2))

        # Plot Training Loss
        plt.subplot(1, 2, 1)  # Subplot in idx [1,1] of [1,2]
        files = sorted([f for f in os.listdir('.') if f.startswith('metrics_batch_') and f.endswith('.pkl')], key=Helper.extract_number)
        colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
                label = f"Batch Size: {file.split('_')[-1].split('.')[0]}"
                plt.semilogy(metrics['training loss'], label=label, alpha=0.9, linewidth=3, color=colors[idx])
        plt.subplots_adjust(left=0.5)
        plt.title('Training Cost Function Minimization Across Batch Size')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)  # Subplot in idx [1,2] of [1,2]
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
                label = f"Batch Size: {file.split('_')[-1].split('.')[0]}"
                plt.semilogy(metrics['training accuracy'], label=label, alpha=0.9, linewidth=3, color=colors[idx])
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        plt.title('Training Accuracy of Model Across Batch Size')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy %')
        plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)
        plt.tight_layout()

    @staticmethod
    def plot_val_metrics():
        plt.figure(figsize=(12+5, 5+2))

        # Plot Training Loss
        plt.subplot(1, 2, 1)  # Subplot in idx [1,1] of [1,2]
        files = sorted([f for f in os.listdir('.') if f.startswith('metrics_batch_') and f.endswith('.pkl')], key=Helper.extract_number)
        colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
                label = f"Batch Size: {file.split('_')[-1].split('.')[0]}"
                plt.semilogy(metrics['validation loss'], label=label, alpha=0.9, linewidth=3, color=colors[idx])
        plt.subplots_adjust(left=0.5)
        plt.title('Validation Cost Function Minimization Across Batch Size')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)  # Subplot in idx [1,2] of [1,2]
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
                label = f"Batch Size: {file.split('_')[-1].split('.')[0]}"
                plt.semilogy(metrics['validation accuracy'], label=label, alpha=0.9, linewidth=3, color=colors[idx])
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        plt.title('Validation Accuracy of Model Across Batch Size')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy %')
        plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)
        plt.tight_layout()

    @staticmethod
    def plot_training_times():
        plt.figure(figsize=(12+5, 5+2))

        plt.subplot(1, 2, 1)  # Subplot in idx [1,1] of [1,2]
        files = sorted([f for f in os.listdir('.') if f.startswith('metrics_batch_') and f.endswith('.pkl')],
                       key=Helper.extract_number)
        colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
        longest_train_time = 0.0

        # First, find the longest training time across all batches
        for file in files:
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
            if 'training time' in metrics:
                total_time = sum(metrics['training time'])
                longest_train_time = max(longest_train_time, total_time)
        # Now use this longest training time to plot all other batch training times
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
            if 'training time' in metrics:
                training_losses = metrics['training loss']
                real_epoch_times = np.cumsum(metrics['training time']) # Plot stops exactly when the training stops, showing a more accurate representation of each batch's training duration
                plt.semilogy(real_epoch_times, training_losses, label=f'Batch Size: {file.split("_")[-1].split(".")[0]}', color=colors[idx], linewidth=3, alpha=0.9)
                # Mark the last point with a distinctive style
                if len(real_epoch_times) > 0:
                    plt.plot(real_epoch_times[-1], training_losses[-1], 'ro', markersize=8, markeredgewidth=1,
                             markeredgecolor='r', markerfacecolor='none')

        plt.xlabel('Understanding Training by Batch Size Total Time (s)') #Total Training Time for Designated Num Epochs (seconds)
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Time by Batch Size')
        plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)

        plt.subplot(1, 2, 2)  # Subplot in idx [1,2] of [1,2]
        for idx, file in enumerate(files):
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
            training_losses = metrics['training loss']
            epochs = list(range(1, len(training_losses) + 1))
            plt.semilogy(epochs, training_losses, label=f'Batch Size: {file.split("_")[-1].split(".")[0]}', color=colors[idx], linewidth=3, alpha=0.9)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Batch Size Impact on Num Epochs Needed to Minimize Cost')
        plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize='medium')
        plt.grid(True)

        plt.tight_layout()  # No overlaps
        plt.show()

    def project(self, config, epochs):
        self.train_model(epochs=max_epochs)
        self.save_training_metrics(self.config.batch_size)
        with open(f'metrics_batch_{self.config.batch_size}.pkl', 'rb') as f:
            loaded_metrics = pickle.load(f)

    def reset_model_and_optimizer(self):
        self.net = Net(self.config)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        return self.net, self.optimizer


# Initialize the network, loss function, optimizer, and objects for tuning and tracking
max_epochs = 3000
Helper.set_seed(seed_value=42)
torch.manual_seed(42)
config = Config()
net = Net(config)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Default step size of 0.001
trainer = Trainer(net, config, criterion, optimizer)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #TODO: Implement LR scheduler!
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10) #get_last_lr()

trainer.project(config, max_epochs) # Default is full batch: for dataset of 3000, this is 2400
batch_sizes = [1200, 600, 64, 32]
for batch_size in batch_sizes:
    config.update_batch_size(batch_size) # Now test on mini-batches of varying size
    trainer.reset_model_and_optimizer()
    trainer.train_model(max_epochs)

Trainer.plot_metrics()
Trainer.plot_val_metrics()
Trainer.plot_training_times()
