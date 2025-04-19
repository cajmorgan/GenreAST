import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.signal import resample_poly

from constants.data import SAMPLE_RATE_TARGET

SUPPORTED_EXTS = ('.wav', '.flac', '.ogg', '.aiff',
                  '.au', '.mp3', '.m4a', '.wma')
DATA_SPLITS = ["train", "val", "test"]


def extract_labels(audio_path: Path):
    label_dirs = [d for d in audio_path.iterdir() if d.is_dir()]
    label_names, labels_idx = [], []

    for label_dir in label_dirs:
        idx, name = label_dir.name.split("_", 1)
        label_names.append(name)
        labels_idx.append(int(idx))

    label_names = np.array(label_names)
    labels_idx = np.array(labels_idx)

    # Sort based on index
    label_names = label_names[labels_idx.argsort()]
    labels_idx.sort()

    # Ensure 'processed' directory exists
    processed_dir = audio_path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save labels
    df = pd.DataFrame({
        "mid": label_names,
        "display_name": label_names
    }, index=labels_idx)

    df.to_csv(processed_dir / "class_labels_indices.csv")
    return label_names


def create_samples(samples, label, path, data_split, data, remove_intros_outros=True):
    out_dir = path.parent.parent / "processed" / data_split
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(samples, desc=f"Processing {data_split} files for label '{label}'"):
        if not file.lower().endswith(SUPPORTED_EXTS):
            continue

        file_path = path / file
        try:
            wav, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        segment_len = 10 * int(sr)
        segments = [
            wav[i:i + segment_len]
            for i in range(0, len(wav) - segment_len + 1, segment_len)
        ]

        if remove_intros_outros:
            segments = segments[3:-3]

        for idx, segment in enumerate(segments):
            if idx in (0, 1):  # Skip early segments
                continue

            safe_filename = re.sub(r"[^A-Za-z0-9\.]", "", file).lower()
            output_path = out_dir / f"{idx}-{safe_filename}"
            sample_json = {
                "wav": str(output_path),
                "labels": label,
                "split": data_split
            }
            data.append(sample_json)

            resampled = resample_poly(segment, SAMPLE_RATE_TARGET, sr)
            sf.write(output_path, resampled, SAMPLE_RATE_TARGET)


def process_audio(audio_path: Path):
    label_names = extract_labels(audio_path)
    data = []

    for root, dirs, files in os.walk(audio_path):
        root_path = Path(root)
        if not files:
            continue

        label = ""
        for name in label_names:
            if name in root_path.name:
                label = name
                break

        files = [f for f in files if f.lower().endswith(SUPPORTED_EXTS)]
        if not files:
            continue

        np.random.shuffle(files)
        split_prop = 0.1
        n_files = len(files)
        split_size = int(np.round(n_files * split_prop))

        test_split = files[:split_size]
        val_split = files[split_size:2 * split_size]
        train_split = files[2 * split_size:]

        create_samples(test_split, label, root_path, "test", data)
        create_samples(val_split, label, root_path, "val", data)
        create_samples(train_split, label, root_path, "train", data)

    # Save dataset summary
    output_path = audio_path.parent / "processed" / "train_test_val_data.json"
    with open(output_path, "w") as f:
        json.dump({"data": data}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio data for training.")
    parser.add_argument("audio_path", type=str,
                        help="Path to the raw audio directory")
    args = parser.parse_args()

    audio_path = Path(args.audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Provided path does not exist: {audio_path}")

    process_audio(audio_path)
