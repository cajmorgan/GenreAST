from datetime import datetime
import math
import librosa
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import ASTFeatureExtractor
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score
from constants.data import MEAN_NORM, SAMPLE_RATE_TARGET, STD_NORM
from model.train_model import train_model
from data.AudioDataset import create_dataloaders
from model.create_ast_model import create_ast_model
from constants.model import BATCH_SIZE, EPOCHS, LEARNING_RATE

try:
    from rich import print
except ImportError:
    pass

# ====== Setup ======
print("[bold green]Setting up environment...[/bold green]")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"[blue]Using device:[/blue] {device}")

# ====== Load Feature Extractor ======
feature_extractor = ASTFeatureExtractor(
    mean=MEAN_NORM,
    std=STD_NORM,
    do_normalize=True
)

# ====== Load & Segment Audio ======
print("[bold green]Loading audio file...[/bold green]")
wav, sr = librosa.load("../materials/example.mp3", sr=44100)
segment_len = 10 * int(sr)

segments = [
    wav[i:i + segment_len]
    for i in range(0, len(wav) - segment_len + 1, segment_len)
]

segment = segments[len(segments) // 2]
resampled = resample_poly(segment, SAMPLE_RATE_TARGET, sr)

# ====== Extract Features ======
print("[bold green]Extracting features...[/bold green]")
features = feature_extractor(
    resampled,
    sampling_rate=SAMPLE_RATE_TARGET,
    return_tensors="pt"
)["input_values"].to(device)

# ====== Load Model ======
print("[bold green]Loading model weights...[/bold green]")
model = create_ast_model(
    num_classes=15,
    device=device,
    weights_path="../model/weights/weights_0.99_map.pth"
)

# ====== Predict ======
print("[bold green]Running prediction...[/bold green]")
prediction = model(features)
predicted_class = prediction.logits.argmax().item()

print(f"[bold yellow]Predicted Class:[/bold yellow] {predicted_class}")
