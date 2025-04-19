

from datetime import datetime
import math
import librosa
import torch
from torch.nn import CrossEntropyLoss
from constants.data import MEAN_NORM, SAMPLE_RATE_TARGET, STD_NORM
from model.train_model import train_model
from data.AudioDataset import create_dataloaders
from model.create_ast_model import create_ast_model
from constants.model import BATCH_SIZE, EPOCHS, LEARNING_RATE
from sklearn.metrics import average_precision_score
import numpy as np
from transformers import ASTFeatureExtractor
from scipy.signal import resample_poly

# Constants
mean_norm = MEAN_NORM
std_norm = STD_NORM
device = "cpu"

# Set device to CUDA or MPS if available
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

feature_extractor = ASTFeatureExtractor(
    mean=mean_norm,
    std=std_norm, do_normalize=True)

# Extract number of classes
num_classes = 15

wav, sr = librosa.load("./example.mp3", sr=44100)

segment_len = 10 * int(sr)
segments = [
    wav[i:i + segment_len]
    for i in range(0, len(wav) - segment_len + 1, segment_len)
]

segment = segments[len(segments) // 2]
resampled = resample_poly(segment, SAMPLE_RATE_TARGET, sr)


features = feature_extractor(
    resampled, sampling_rate=SAMPLE_RATE_TARGET, return_tensors="pt"
)["input_values"]


features = features.to(device)
# Load the model
model = create_ast_model(num_classes=num_classes, device=device,
                         weights_path="../model/weights/weights_0.99_map.pth")


prediction = model(features)


print(prediction.logits.argmax())
