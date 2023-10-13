from skmultilearn.dataset import load_dataset
import pandas as pd
from tqdm import tqdm

datasets = [
    'Corel5k',
    'genbase',
    'bibtex',
    'birds',
    'delicious',
    'emotions',
    'enron',
    'mediamill',
    'medical',
    'rcv1subset1',
    'rcv1subset2',
    'rcv1subset3',
    'rcv1subset4',
    'rcv1subset5',
    'scene',
    'tmc2007_500',
    'yeast',
]

# Define column names and types
columns = {
    "n_samples": int,
    "n_features": int,
    "n_labels": int,
    "micro_label_density": float,
    "macro_label_density": float,
    "samples_label_density": float,
}

# Create empty DataFrame with column names and types
metadata = pd.DataFrame(columns=columns.keys())

for dataset in tqdm(datasets):
    X, y, _, _ = load_dataset(dataset, "undivided")
    metadata.loc[dataset, :] = (
        X.shape[0],
        X.shape[1],
        y.shape[1],
        y.mean(),
        y.mean(0).mean(),
        y.mean(1).mean(),
    )
metadata = metadata.astype(columns).sort_values("n_samples")
metadata.to_csv("dataset_metadata.tsv", sep="\t")

print(metadata)
print("Saved dataset metadata to dataset_metadata.tsv")