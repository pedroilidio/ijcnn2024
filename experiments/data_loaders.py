from pathlib import Path

import pandas as pd
import numpy as np


def load_nakano(path):
    data = pd.read_csv(path)
    y_cols = data.columns.str.startswith("label")
    return {
        "X": data.loc[:, ~y_cols].to_numpy(),
        "y": data.loc[:, y_cols].to_numpy(),
    }
