from typing import Dict, Iterable, Sequence
import copy
import itertools
import inspect
import logging
import pickle
from pprint import pformat
import warnings
import requests
import yaml
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from hashlib import sha3_256
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
import numpy as np

DEF_PATH_CONFIG = Path("config.yml")
DEF_PATH_LOG = Path("experiments.log")
DEF_LOG_LEVEL = "INFO"
DT_FORMAT = r"%Y%m%dT%H%M%S%f"


class ConfigLoadingError(RuntimeError):
    pass


def convert_iterables_to_lists(data):
    if isinstance(data, str):
        return data
    if isinstance(data, Dict):
        return {k: convert_iterables_to_lists(data[k]) for k in data.keys()}
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, Iterable):
        return [convert_iterables_to_lists(k) for k in data]
    return data


def load_object(class_path: str) -> Any:
    module_path, class_ = class_path.rsplit(".", maxsplit=1)
    return getattr(import_module(module_path), class_)


def load_callable(data: dict) -> Callable:
    # Allow unsafe YAML loader to load objects.
    if not isinstance(data["call"], str):
        return data["call"]

    func = load_object(data["call"])
    return partial(func, **data["params"])


def load_estimator(estimator_data: dict) -> Any:
    """Load estimator object from dict(call=*, params={*})."""
    if isinstance(estimator_data["call"], str):
        estimator = load_object(estimator_data["call"])
    else:
        # Allow unsafe YAML loader to load objects.
        estimator = estimator_data["call"]

    if callable(estimator):
        estimator = estimator(**estimator_data["params"])
    else:
        estimator.set_params(**estimator_data["params"])

    if "search" in estimator_data:
        grid_search_class = load_callable(estimator_data["search"])
        estimator = grid_search_class(estimator=estimator)

    return estimator


def load_table(table_data: dict) -> Any:
    """table_data must contain the keys:

    path (str): filepath where to save to and load from
    url (str): if file not found, attempts to download it from this url
    force_download (bool): download again even if present
    read['call'] (str): module path to the reading function. e.g.
        pandas.read_csv
    read['params'] (dict): params to pass on to loaded read_func
    """
    path_table = Path(table_data["path"])

    # Download file if not found.
    if not path_table.exists() or table_data.get("force_download", False):
        if "url" not in table_data:
            raise FileNotFoundError(
                f"{path_table} does not exist and no source URL was provided."
                " Please check the dataset's location or provide a value to its 'url'"
                " attribute in the configuration file."
            )
        res = requests.get(table_data["url"])
        res.raise_for_status()
        path_table.parent.mkdir(exist_ok=True, parents=True)
        with path_table.open("wb") as f:
            f.write(res.content)

    read_func = load_callable(table_data["read"])
    table = read_func(table_data["path"])
    return table


def load_dataset(dataset: dict) -> dict:
    """Load all tables in the dataset.

    Dataset example:
    ```
    - name: enzymes
      pairwise: true
      X:
        - path: data/enzymes/x1.csv
          url: https://[...]enzymes_x1.csv
        - path: data/enzymes/x2.csv
          url: https://[...]enzymes_x2.csv
      y:
        path: data/enzymes/y_already_here.csv
        read:
          call: pandas.read_csv
          params:
            header: [0, 1]
            index_col: 0
    ```
    """
    if "call" in dataset:
        return (
            dataset | load_callable(dataset)()
        )  # Must return a dict with X and y keys.
    if isinstance(dataset["X"], list):
        X = [load_table(x) for x in dataset["X"]]
    else:
        X = load_table(dataset["X"])

    return dataset | dict(X=X, y=load_table(dataset["y"]))


def deep_update(A: Any, B: Any) -> Any:
    """Apply defaults in A config to B config file.

    It matches A to B recursively. If both are dictionaries, performs a simple
    `dict.update()`. If A is a dict and B is a list, use A to update
    every element in B. If both are lists, return them concatenated.

    Args:
        A (dict, list, scalar): default configurations.
        B (dict, list, scalar): user defined settings.

    Returns:
        (dict): A "deeply updated" by B.
    """
    if not isinstance(B, (list, dict)):  # If we receive scalar values.
        return B

    elif isinstance(A, list):
        assert isinstance(B, list)
        return A + B

    elif isinstance(B, dict):
        ret = copy.deepcopy(A)
        for k, Bk in B.items():
            ret[k] = Bk if k not in A else deep_update(A[k], Bk)
        return ret

    else:  # elif isinstance(B, list):
        return [deep_update(A, item) for item in B]


def omit_not_builtin_params(param) -> Any:
    if isinstance(param, Callable):  # type or function
        return dict(
            load=param.__module__ + "." + param.__name__,
        )
    # If it's an instantiated object
    if type(param).__module__ != "builtins":
        return dict(
            call=type(param).__module__ + "." + type(param).__name__,
            params=(
                omit_not_builtin_params(param.get_params())
                if hasattr(param, "get_params")
                else {}
            ),
        )
    if isinstance(param, Dict):
        return {
            p: omit_not_builtin_params(v)
            for p, v in param.items()
            if not (p.endswith("_") or p.startswith("_"))
        }
    if not isinstance(param, str) and isinstance(param, Sequence):
        return [omit_not_builtin_params(p) for p in param]

    if type(param).__module__ == "builtins":
        return param

    warnings.warn(f"{param} will be replaced by None")
    return None


def compute_run_hash(run: dict) -> str:
    """Compute hash based on static parameters.
    Parameters used should not change each time the run is loaded.
    Parameters
    ----------
    run : dict
        run data
    Returns
    -------
    str
        SHA256 hash in hexadecimal form.
    """
    # Collect attributes that do not impact on CV results and are not expected
    # to differ between identically configured runs.
    estimator_params = omit_not_builtin_params(
        run["estimator"]["final_params"],
    )
    cv_params = inspect.signature(load_callable(run["cv"])).bind_partial(
        **run["cv"]["params"]
    )
    cv_params.apply_defaults()
    cv_params = omit_not_builtin_params(cv_params.arguments)
    if (
        "scoring" in cv_params
        and isinstance(cv_params["scoring"], Sequence)
        and not isinstance(cv_params["scoring"], str)
    ):
        # Ignore the order in which the scorers were provided
        cv_params["scoring"] = list(sorted((cv_params["scoring"])))

    # Ignore parameters that do not affect the results
    for param_to_ignore in (
        "n_jobs",
        "verbose",
        "pre_dispatch",
    ):
        cv_params.pop(param_to_ignore, None)

    parameters_hash = sha3_256()

    for obj in (
        estimator_params,
        cv_params,
        run["dataset"],
    ):
        parameters_hash.update(pickle.dumps(obj))

    return parameters_hash.hexdigest()


def get_run_filename(run: dict) -> str:
    """Compute run filename.

    Args:
        run (dict): dictionary containing run information.

    Returns:
        (str): run filename.
    """
    return (
        f"{run['hash'][:7]}_{datetime.strftime(run['start'], DT_FORMAT)}"
        f"_{run['estimator']['name']}_{run['dataset']['name']}.yml"
    )


def execute_run(
    run: dict,
    allow_redundant_runs: bool = False,
) -> dict | None:
    """Execute run instructions.

    Args:
        run (dict): dictionary containing run information.
    """
    run = copy.deepcopy(run)
    run["start"] = datetime.now()

    cv_data = run["cv"].copy()
    cv_function = load_callable(cv_data)
    dataset = load_dataset(run["dataset"])
    estimator = load_estimator(run["estimator"])

    # FIXME: There must be a better way to do this.
    if "modify_params" in run:
        estimator.set_params(**run["modify_params"])
        param_suffix = "_".join(str(p) for p in run["modify_params"].values())
        run["estimator"]["name"] += "__" + param_suffix
    if "wrapper" in run and run["wrapper"] is not None:
        estimator = load_callable(run["wrapper"])(estimator)
        run["estimator"]["name"] += "__" + run["wrapper"]["name"]

    run["estimator"]["final_params"] = (
        omit_not_builtin_params(estimator.get_params(deep=False))
        if hasattr(estimator, "get_params")
        else {}
    )

    run["hash"] = compute_run_hash(run)

    outdir = Path(run["directory"])
    run["path"] = str((outdir / get_run_filename(run)).resolve())

    if not allow_redundant_runs:
        try:
            # FIXME: current hashes need to be recalculated.
            # next(outdir.rglob(f"{run['hash'][:7]}*.yml"))
            next(
                outdir.rglob(
                    f"*{run['estimator']['name']}_{run['dataset']['name']}*.yml"
                )
            )
        except StopIteration:
            pass
        else:
            logging.warning(
                "A run with same parameters (estimator="
                f"{run['estimator']['name']}, dataset={run['dataset']['name']}"
                f", hash={run['hash'][:7]}) is already "
                f"present in {outdir}. Pass --allow-redundant-runs if you "
                "intend to keep all of them."
            )
            return None

    logging.info(f"Starting run with parameters:\n{pformat(run)}")

    # Execute run
    run["results"] = convert_iterables_to_lists(
        cv_function(
            estimator=estimator,
            X=dataset["X"],
            y=dataset["y"],
        )
    )
    run["end"] = datetime.now()

    logging.info(f"Run {run['hash']} finished without errors.")
    return run


def load_runs(
    config_file: Path = DEF_PATH_CONFIG,
    unsafe_yaml: bool = False,
) -> dict:
    """Properly load YAML configuration files.

    Args:
        config_file (Path): main file with three main keys: databases,
            estimators and runs. Each one's format is specified in its loading
            function.
        defaults_file (Path): settings to apply if not specified in
            config_file.
        unsafe_yaml (bool): whether or not to use `yaml.UnsafeLoader`.

    Yields:
        run (dict): each run informations to be processed by `execute_run()`.
    """
    yaml_load = yaml.unsafe_load if unsafe_yaml else yaml.safe_load

    try:
        with config_file.open() as cf:
            config = yaml_load(cf)
    except yaml.constructor.ConstructorError:
        raise ConfigLoadingError(
            "It seems that your config files tried to load a Python object. If"
            " you know what you are doing, you could try enabling YAML unsafe "
            "loader with the --unsafe-yaml option. Proceed with caution."
        )

    config = deep_update(config["defaults"], config)
    aliases = {
        kind: {d["name"]: d for d in config["aliases"][kind]}
        for kind in config["aliases"]
    }

    for run in config["runs"]:
        if not run["active"]:
            logging.info(f"Skipping inactive run:\n{yaml.dump(run)}")
            continue

        # Set up for using itertools.product in the next step
        for key, value in run.items():
            if not isinstance(value, list):
                run[key] = [value]

        # If multiple values provided, make a run for each option combination.
        for values in itertools.product(*run.values()):
            subrun = dict(zip(run.keys(), values))

            # Parse string references
            for key, value in subrun.items():
                if isinstance(value, str) and (key in aliases):
                    subrun[key] = aliases[key][value]

            yield subrun


# TODO: Arguments description
def make_argparser(default_args: dict | None = None) -> ArgumentParser:
    default_args = default_args or {}
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-c", default=DEF_PATH_CONFIG, type=Path)
    parser.add_argument("--log-file", "-l", default=DEF_PATH_LOG, type=Path)
    parser.add_argument("--log-level", "-L", default=DEF_LOG_LEVEL)
    parser.add_argument("--unsafe-yaml", action="store_true")
    parser.add_argument(
        "--allow-redundant-runs",
        action="store_true",
        help=(
            "Allow running multiple runs with the same set of parameters. The"
            "default behaviour is to skip a run if an equivalent record is "
            "found in the output directory."
        ),
    )
    parser.add_argument(
        "--raise-errors",
        action="store_true",
        help=(
            "Stop excecution and raise exception if an error occur. The "
            "default behaviour is to log the occurence and continue to the "
            "next run."
        ),
    )

    parser.set_defaults(**default_args)
    return parser


def main(
    config_file: Path = DEF_PATH_CONFIG,
    log_file: Path = DEF_PATH_LOG,
    log_level: str = DEF_LOG_LEVEL,
    unsafe_yaml: bool = False,
    raise_errors: bool = False,
    allow_redundant_runs: bool = False,
) -> None:
    logging.captureWarnings(True)
    logging.basicConfig(
        level=log_level,
        format="* [%(levelname)s] %(asctime)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Starting a new series of runs.")

    for run in load_runs(config_file, unsafe_yaml):
        try:
            finished_run = execute_run(run, allow_redundant_runs)
        except KeyboardInterrupt:
            logging.warning("Run terminated by user (KeyboardInterrupt).")
            raise
        except Exception:
            logging.exception("Run terminated with error.")
            if raise_errors:
                raise
        else:
            if finished_run is None:
                continue
            run_path = Path(finished_run["path"])
            run_path.parent.mkdir(exist_ok=True, parents=True)

            with run_path.open("w") as out:
                yaml.dump(finished_run, out)
            logging.info(f"Run information saved to {run_path}.")


if __name__ == "__main__":
    args = make_argparser().parse_args()
    main(**vars(args))
