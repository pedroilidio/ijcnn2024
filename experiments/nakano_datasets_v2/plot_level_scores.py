import sys
import warnings

import joblib
import yaml
import matplotlib.pyplot as plt
from sklearn.base import clone
from imblearn.pipeline import Pipeline

from deep_forest.cascade import Cascade
from estimators import scoring_metrics
sys.path.insert(0, "..")
from run_experiments import load_dataset
# from positive_dropper import PositiveDropper


path_metadata = "runs/0bab2e5_20231021T211052729643_embedding_weak_chi2_genbase.yml"
path_model = "runs/0bab2e5_20231021T211052729643_embedding_weak_chi2_genbase.joblib"

print("Loading metadata...")
with open(path_metadata, "r") as file_metadata:
    metadata = yaml.unsafe_load(file_metadata)

print("Loading model...")
models = joblib.load(path_model)
dataset = load_dataset(metadata["dataset"])
cv = metadata["cv"]["params"]["cv"]

scores = []
for model, (train_idx, test_idx) in zip(models, cv.split(dataset["X"], dataset["y"])):
    X_train, y_train = dataset["X"][train_idx], dataset["y"][train_idx]
    X_test, y_test = dataset["X"][test_idx], dataset["y"][test_idx]

    if (
        not isinstance(model, Pipeline)  # If not wrapped with positive dropper
        and not isinstance(model["estimator"], Cascade)
    ):
        warnings.warn("Skipping model that is not a cascade.")
        continue

    if not isinstance(model, Cascade):  # isinstance(model, Pipeline):
        _, y_train_t = model["dropper"].fit_resample(None, y_train)
        model = model["estimator"]
    else:
        y_train_t = y_train

    breakpoint()
    model._apply_transformers_and_samplers
    final_estimator = clone(model.final_estimator)
    # Get the scores on each level of the Cascade object
    scores.append(model.predict())

# Plot the scores on each level
plt.plot(scores)
plt.xlabel("Level")
plt.ylabel("Score")
plt.title("Scores on Each Level")
plt.show()
