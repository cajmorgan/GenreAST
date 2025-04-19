import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor
from constants.data import MEAN_NORM, SAMPLE_RATE_TARGET, STD_NORM


class AudioDataset(Dataset):
    def __init__(self, json_file, labels_file, split="train", oversampling=None, oversampling_exponent=0,
                 mean_norm=0, std_norm=0, normalize=True):

        self.feature_extractor = ASTFeatureExtractor(
            mean=mean_norm,
            std=std_norm, do_normalize=normalize)
        self.oversampling_exponent = oversampling_exponent

        with open(json_file, "r") as f:
            raw_data = json.load(f)["data"]

        self.labels = pd.read_csv(labels_file)["mid"].tolist()
        self.label_to_idx = {label: idx for idx,
                             label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label,
                             idx in self.label_to_idx.items()}

        df = pd.DataFrame(raw_data)
        self.data = df[df["split"] == split].reset_index(drop=True)

        if oversampling == "random_oversampling":
            self._apply_random_oversampling()

    def _apply_random_oversampling(self):
        prios = self.data["labels"].value_counts(
        ) / self.data["labels"].value_counts().sum()
        prios_vals = np.asarray(prios.values)
        prios_labels = prios.index.values

        max_prio = prios_vals[0]
        prios_expo = prios_vals ** self.oversampling_exponent

        # How many times we repeat each sample in a class
        M = np.ceil(max_prio / prios_expo)

        new_dfs = []
        for idx, m in enumerate(M):
            current_label = prios_labels[idx]
            df_filtered_by_label = (
                self.data[self.data.labels == current_label])

            df_repeated = df_filtered_by_label.loc[df_filtered_by_label.index.repeat(
                m)].reset_index(drop=True)
            new_dfs.append(df_repeated)

        self.data = pd.concat(new_dfs, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        wav_path = Path(entry["wav"].replace("./../", "./"))

        wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE_TARGET)
        label_idx = self.label_to_idx[entry["labels"]]

        features = self.feature_extractor(
            wav, sampling_rate=SAMPLE_RATE_TARGET, return_tensors="pt"
        )["input_values"][0]

        return features, label_idx


def create_dataloaders(json_path, labels_path, mean_norm, std_norm, batch_size=16, normalize=True):
    ds_train = AudioDataset(json_path, labels_path, split="train",
                            oversampling="random_oversampling",  mean_norm=mean_norm, std_norm=std_norm, oversampling_exponent=0, normalize=normalize)
    ds_val = AudioDataset(json_path, labels_path,
                          split="val", mean_norm=mean_norm, std_norm=std_norm, normalize=normalize)
    ds_test = AudioDataset(json_path, labels_path,
                           split="test", mean_norm=mean_norm, std_norm=std_norm, normalize=normalize)

    return {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(ds_val, batch_size=1, shuffle=False),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False),
        "num_classes": len(ds_train.labels)
    }


# Example usage
if __name__ == "__main__":
    dataloaders = create_dataloaders(
        json_path="./../data/processed/train_test_val_data.json",
        labels_path="./../data/processed/class_labels_indices.csv",
        mean_norm=MEAN_NORM,
        std_norm=STD_NORM
    )

    for batch in dataloaders["train"]:
        print(batch)
        break
