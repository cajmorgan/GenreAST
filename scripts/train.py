
from datetime import datetime
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from constants.data import MEAN_NORM, STD_NORM
from model.train_model import train_model
from data.AudioDataset import create_dataloaders
from model.create_ast_model import create_ast_model
from constants.model import BATCH_SIZE, EPOCHS, LEARNING_RATE


mean_norm = MEAN_NORM
std_norm = STD_NORM

device = "cpu"
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

dataloaders = create_dataloaders(
    json_path="./../data/processed/train_test_val_data.json",
    labels_path="./../data/processed/class_labels_indices.csv",
    mean_norm=mean_norm,
    std_norm=std_norm,
    normalize=True,
    batch_size=BATCH_SIZE,

)

num_classes = dataloaders["num_classes"]

# Assuming your model is already loaded and moved to the device
model = create_ast_model(num_classes=num_classes, device=device,
                         weights_path="../model/weights/weights_0.99_map.pth")
# # Optimizer and loss function
optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss()

# Mixed precision scaler
scaler = GradScaler()

# # Train the model
train_model(model, dataloaders["train"], dataloaders["val"], optimizer,
            loss_fn, scaler, device, epochs=EPOCHS, accumulation_steps=16)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
torch.save(model.state_dict(), f'../model/weights/weights_{timestamp}.pth')
