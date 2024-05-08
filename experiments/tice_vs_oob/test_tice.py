import time
import sys

from bitarray import bitarray
import numpy as np

# import tice
sys.path.insert(0, "SAR-PU/lib/km")
import km

# sys.path.insert(0, "SAR-PU/sarpu")
from sarpu.evaluation import *
from sarpu.input_output import *
from sarpu.labeling_mechanisms import parse_labeling_model
from sarpu.paths_and_names import *
from sarpu.pu_learning import *
from sarpu.labeling_mechanisms import label_data
from sarpu.paths_and_names import *
from sarpu.experiments import *


def test_tice():
    data_folder= "SAR-PU/Data/"
    results_folder="SAR-PU/Results/"

    data_name = "mushroom_extclustering"
    # data_name = "mushroom"
    propensity_attributes = [111,112,113,114]
    propensity_attributes_signs = [1,1,1,1]
    settings = "lr._.lr._.0-111"
    labeling_model_type = "simple_0.2_0.8"

    labeling=0
    partition=1

    nb_assignments=5
    nb_labelings=5

    relabel_data = False
    rerun_experiments = False

    labeling_model = label_data(
        data_folder, 
        data_name, 
        labeling_model_type, 
        propensity_attributes, 
        propensity_attributes_signs, 
        nb_assignments,
        relabel_data=relabel_data
    )

    # Load data
    x_path = data_path(data_folder,data_name)
    y_path = classlabels_path(data_folder,data_name)
    s_path = propensity_labeling_path(data_folder, data_name, labeling_model, labeling)
    e_path = propensity_scores_path(data_folder, data_name, labeling_model)
    f_path = partition_path(data_folder, data_name, partition)
    (x_train,y_train,s_train,e_train),(x_test,y_test,s_test,e_test) = read_data((x_path,y_path,s_path,e_path),f_path)

    for drop in (0.2, 0.4, 0.5, 0.7):
        positive_indices = np.where(s_train == 1)[0]
        drop_indices = np.random.choice(
            positive_indices,
            int(drop * len(positive_indices)),
            replace=False,
        )

        s_train_corrupted = s_train.copy()
        s_train_corrupted[drop_indices] = 0

        x_train_tice = (x_train + 1) / 2
        s_train_tice = bitarray(list(s_train_corrupted == 1))
        tice_folds = np.random.randint(5, size=len(s_train))

        start = time.time()
        (c_est, all_c_est) = tice(x_train_tice, s_train_tice, 5, tice_folds)
        time_c = time.time() - start
        print(f"tice: {time_c=}, mean={s_train_corrupted.mean()} {c_est=}, {all_c_est=}")


def main():
    test_tice()


if __name__ == "__main__":
    main()