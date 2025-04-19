from data.AudioDataset import create_dataloaders
from torch import GradScaler, autocast
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm  # Importing tqdm for progress bars


def train_model(model, train_loader, val_loader, optimizer, loss_fn, scaler, device, epochs=10, accumulation_steps=16):
    """
    Train the AST model using gradient accumulation with mixed precision.
    This version includes a progress bar for training steps and epochs.
    """
    print("Training init...")

    model.train()
    total_train_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        # Create a progress bar for training
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            for idx, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                # Mixed precision forward pass
                with autocast(device_type=device):
                    outputs = model(features)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)

                # Backpropagation with gradient scaling
                scaler.scale(loss).backward()

                if (idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item()

                if (idx + 1) % accumulation_steps == 0:
                    avg_loss = epoch_loss / accumulation_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    total_train_loss.append(avg_loss)
                    epoch_loss = 0

                # Update progress bar
                pbar.update(1)

        # Validation after each epoch
        validate_model(model, val_loader, loss_fn, device)


def validate_model(model, val_loader, loss_fn, device):
    """
    Validation step after each epoch to evaluate the model performance.
    """
    model.eval()
    val_loss = []

    with torch.no_grad():
        # Create a progress bar for validation
        with tqdm(total=len(val_loader), desc="Validation", unit="step") as pbar:
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                with autocast(device_type=device):
                    outputs = model(features)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    val_loss.append(loss.item())

                # Update validation progress bar
                pbar.update(1)

    avg_val_loss = np.mean(val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    model.train()  # Set model back to training mode
