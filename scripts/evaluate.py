from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from constants.data import MEAN_NORM, STD_NORM
from model.train_model import train_model
from data.AudioDataset import create_dataloaders
from model.create_ast_model import create_ast_model
from constants.model import BATCH_SIZE, EPOCHS, LEARNING_RATE
from sklearn.metrics import average_precision_score
import numpy as np

# Constants
mean_norm = MEAN_NORM
std_norm = STD_NORM
device = "cpu"

# Set device to CUDA or MPS if available
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

# Create dataloaders
dataloaders = create_dataloaders(
    json_path="./../data/processed/train_test_val_data.json",
    labels_path="./../data/processed/class_labels_indices.csv",
    mean_norm=mean_norm,
    std_norm=std_norm,
    normalize=True,
    batch_size=BATCH_SIZE,
)

# Extract number of classes
num_classes = dataloaders["num_classes"]

# Load the model
model = create_ast_model(num_classes=num_classes, device=device,
                         weights_path="../model/weights/weights_0.99_map.pth")

# Initialize variables for evaluation
true_pos = 0
tot = 0
scores = []
labels = []
y_true = []
y_pred = []

# Disable gradient tracking for evaluation
torch.set_printoptions(sci_mode=False)

# Perform evaluation
with torch.no_grad():
    for sample_idx, (sample, label) in enumerate(dataloaders["test"]):
        print(sample_idx)
        # Move sample to the correct device
        sample = sample.to(device)

        # Get model predictions
        out = model(sample).logits
        probs = torch.nn.functional.softmax(out, dim=-1)

        label = label.cpu().item()
        # Append scores and labels
        probs = (probs.cpu().detach().numpy())[0]
        scores.append(probs.tolist())
        labels.append(label)

        # Get predicted and true labels
        pred = probs.argmax().item()
        true = label
        y_true.append(true)
        y_pred.append(pred)

        # Update true positives and total count
        true_pos += int(pred == true)
        tot += 1


# Calculate and print metrics
all_AP = average_precision_score(labels, scores, average="macro")
print("Mean Average Precision (mAP):", np.mean(all_AP))
print(f"True Positives: {true_pos}")
print(f"Total Samples: {tot}")
print(f"Accuracy: {true_pos / tot:.4f}")
