from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
OUTPUT = Path("progress_table.tsv")


def load_estimator_dataset(path):
    with open(path, "r") as f:
        run = yaml.unsafe_load(f)
    return run["estimator"]["name"], run["dataset"]["name"]


print("Inspecting runs...")
df = pd.DataFrame.from_records(
    [load_estimator_dataset(p) for p in tqdm(list(RUNS_DIR.glob("*.yml")))],
    columns=["model", "dataset"],
)
crosstab = pd.crosstab(df["dataset"], df["model"])
crosstab.to_csv(OUTPUT, sep="\t")
print(f"Saved progress table to {OUTPUT}.")

# Save image
plt.figure(figsize=[i/2.36 + 2 for i in crosstab.T.shape])
sns.heatmap(crosstab, annot=True, square=True, fmt="d", cbar=False)
plt.tight_layout()
plt.savefig("progress_table.png")
print(f"Saved progress table image to progress_table.png.")
