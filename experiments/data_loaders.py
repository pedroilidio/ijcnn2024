from pathlib import Path
import warnings

import pandas as pd
import numpy as np


def load_nakano(path, min_positives=0):
    data = pd.read_csv(path)
    y_cols = data.columns.str.startswith("label")
    X = data.loc[:, ~y_cols].to_numpy()
    y = data.loc[:, y_cols].to_numpy()

    if isinstance(min_positives, float):
        min_positives =  int(np.ceil(min_positives * y.shape[0]))
    if min_positives:
        warnings.warn(
            f"Label columns {np.where(y.sum(axis=0) > min_positives)[0].tolist()}"
            f" of the dataset (which had {y.shape[0]} label columns in total) have"
            f" less than {min_positives} positives. Dropping them during loading."
        )
        y = y[:, y.sum(axis=0) >= min_positives]

        if y.shape[1] == 0:
            raise ValueError(
                f"None of the label columns of the dataset had more than"
                f" {min_positives} positives."
            )
    return {"X": X, "y": y}
