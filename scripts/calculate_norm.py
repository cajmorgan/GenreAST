import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

from data.AudioDataset import AudioDataset, create_dataloaders


def calculate_norm(data_loader: DataLoader):
    mean = []
    std = []

    # Use tqdm to add a progress bar
    with tqdm(total=len(data_loader), desc="Calculating Mean and Std", ncols=100) as pbar:
        for i, (audio_input, labels) in enumerate(data_loader):
            cur_mean = torch.mean(audio_input)
            cur_std = torch.std(audio_input)
            mean.append(cur_mean)
            std.append(cur_std)

            # Update the progress bar without printing every batch result
            # Add mean and std to progress bar
            pbar.set_postfix(mean=cur_mean.item(), std=cur_std.item())
            pbar.update(1)  # Update progress bar by 1 step

    # Calculate and print the overall mean and std after processing all batches
    overall_mean = np.mean(mean)
    overall_std = np.mean(std)
    print(
        f"\nFinished processing all batches.\nOverall Mean = {overall_mean:.4f}, Overall Std = {overall_std:.4f}")


# Assuming create_dataloaders is defined correctly
dataloaders = create_dataloaders(
    json_path="./../data/processed/train_test_val_data.json",
    labels_path="./../data/processed/class_labels_indices.csv",
    mean_norm=0,
    std_norm=0,
    normalize=False
)

calculate_norm(dataloaders["train"])
