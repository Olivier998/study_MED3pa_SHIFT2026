"""
MED3pa + Detectron experiments under controlled covariate shifts.

For each task in TASKS and each exclusion criterion in EXCLUSION_CRITERIA
(applied one at a time), the following pipeline is run:

    1. Load and clean datasets/{task}.csv, one-hot encode the categorical
       columns (encoder fitted on the full dataset).
    2. Split the dataset in two random halves (fixed seed, identical for
       every criterion).
    3. Induce the covariate shift: in the first half (training set) only,
       keep the rows satisfying the criterion (e.g. age > 50).
    4. Fit an iterative imputer and a standard scaler on the training set,
       and apply them to both halves.
    5. Tune (Optuna, 5-fold CV balanced accuracy) and train a PyTorch MLP
       base model on the training set.
    6. Run MED3pa on the second half (evaluation set), with the Detectron
       and the FeatureResumer explaining each profile.
    7. Save the base model, its metrics, the MED3pa results, the MDR curves
       and the profile tree under experiments/results/{task}/{criterion}/.

Fitted encoders, imputers, scalers and base models are cached in the
results folder and reused on later runs.

Once the profile trees are generated, `experiments/extract_profiles.py`
collects the profiles in a CSV, and running this script again with
`--profile-metrics <csv>` computes the base model metrics on every profile,
for all evaluation patients and separately for those excluded from /
included in the training population.

Usage (from the repository root):
    python -m experiments.main
    python -m experiments.extract_profiles
    python -m experiments.main --profile-metrics experiments/results/profiles.csv
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MED3pa.datasets import DatasetsManager
from MED3pa.med3pa import Med3paExperiment
from MED3pa.models import BaseModelManager
from MED3pa.uncertainty_explainer.covariate_shit.detectron import Detectron
from MED3pa.uncertainty_explainer.covariate_shit.feature_resumer import FeatureResumer
from MED3pa.uncertainty_explainer.explainer import Explainer
from MED3pa.visualization.mdr_visualization import visualize_mdr
from MED3pa.visualization.profiles_visualization import visualize_tree

from src.metrics import METRIC_NAMES, compute_metrics
from src.mlp import TorchMLPClassifier, tune_mlp
from src.preprocessing import (apply_criterion, clean_dataset, criterion_mask, encode_categoricals,
                               fit_categorical_encoder, infer_categorical_columns, profile_mask)
from src.profiles import load_profile_paths, parse_profile

logging.getLogger("ray").setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TARGET_COL = "target"
N_OPTUNA_TRIALS = 50
RESULTS_DIR = "experiments/results"

TASKS = {
    "30dr": "datasets/30dr.csv",
    "mortality_48h": "datasets/mortality_48h.csv",
    "disch_alive_48h": "datasets/disch_alive_48h.csv",
}

# Rows of the training set satisfying the criterion are kept, the others are
# excluded. Each criterion is applied on its own (one experiment per criterion).
EXCLUSION_CRITERIA = [
    {"name": "age_gr_50", "column": "age", "operator": ">", "value": 50},
    {"name": "sex_m_eq_1", "column": "gender_M", "operator": "==", "value": 1},
    {"name": "no_diabetes", "column": "Diabetes mellitus", "operator": "==", "value": 0},
    {"name": "race_WHITE_eq_1", "column": "race_WHITE", "operator": "==", "value": 1},
]
# The diabetes indicator is only available for the 30-day readmission task
DIABETES_TASKS = ["30dr"]

IPC_PARAMS = {"n_estimators": 100}
IPC_GRID = {
    "n_estimators": [100, 200, 350, 500],
    "max_depth": [None, 6, 12, 18, 24],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", None],
}
APC_PARAMS = {"max_depth": 6}
APC_GRID = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_leaf": [5, 10],
}


def load_or_fit(path: str, fit, name: str):
    """Load a pickled object if `path` exists, otherwise build it with `fit()` and pickle it."""
    if os.path.exists(path):
        print(f"Loading existing {name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"Fitting {name}")
    obj = fit()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def run_experiment(task_name: str, csv_path: str, criterion: dict, results_dir: str,
                   n_rows: int | None = None, profile_paths: list | None = None) -> None:
    run_label = f"{task_name}/{criterion['name']}"
    print(f"\n=== Running: {run_label} ===")
    out_dir = f"{results_dir}/{task_name}/{criterion['name']}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load, clean and encode
    df = pd.read_csv(csv_path)
    if n_rows is not None:
        df = df.head(n_rows)
    df = clean_dataset(df, TARGET_COL)

    categorical_columns = infer_categorical_columns(df, TARGET_COL)
    encoder = load_or_fit(f"{out_dir}/categorical_encoder.pkl",
                          lambda: fit_categorical_encoder(df, categorical_columns), "categorical encoder")
    df = encode_categoricals(df, categorical_columns, encoder)

    # 2. Random split in two halves
    first_half, second_half = train_test_split(df, test_size=0.5, random_state=SEED)
    first_half = first_half.reset_index(drop=True)
    second_half = second_half.reset_index(drop=True)
    n_first_half_before = len(first_half)

    # 3. Induce the shift in the training set only
    first_half = apply_criterion(first_half, criterion)
    n_first_half_after = len(first_half)
    # Evaluation patients belonging to the training population
    second_half_included = criterion_mask(second_half, criterion).to_numpy()

    # The diabetes indicator is constant in the shifted training set: drop it
    if criterion["name"] == "no_diabetes":
        first_half = first_half.drop(columns=[criterion["column"]])
        second_half = second_half.drop(columns=[criterion["column"]])

    # 4. Imputation and standardization, fitted on the training set
    feature_columns = [c for c in first_half.columns if c != TARGET_COL]

    def fit_imputer():
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=20, n_jobs=-1, random_state=SEED),
            max_iter=5,
            random_state=SEED,
        )
        return imputer.fit(first_half[feature_columns])

    imputer = load_or_fit(f"{out_dir}/imputer.pkl", fit_imputer, "imputer")
    first_half_x = pd.DataFrame(imputer.transform(first_half[feature_columns]), columns=feature_columns)
    second_half_x = pd.DataFrame(imputer.transform(second_half[feature_columns]), columns=feature_columns)
    first_half_y = first_half[TARGET_COL].to_numpy()
    second_half_y = second_half[TARGET_COL].to_numpy()

    scaler = load_or_fit(f"{out_dir}/scaler.pkl", lambda: StandardScaler().fit(first_half_x), "scaler")
    first_half_x = pd.DataFrame(scaler.transform(first_half_x), columns=feature_columns)
    second_half_x = pd.DataFrame(scaler.transform(second_half_x), columns=feature_columns)

    # 5. Base model
    model_path = f"{out_dir}/base_model.pkl"
    params_path = f"{out_dir}/mlp_best_params.json"
    study = None
    if os.path.exists(model_path):
        print("Loading existing base model")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        best_params = None
        if os.path.exists(params_path):
            with open(params_path) as f:
                best_params = json.load(f)
    else:
        print("Tuning MLP hyperparameters with Optuna")
        best_params, study = tune_mlp(first_half_x, first_half_y, n_trials=N_OPTUNA_TRIALS, seed=SEED)

        print("Training base model")
        clf = TorchMLPClassifier(
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            activation=best_params["activation"],
            lr=best_params["learning_rate_init"],
            max_epochs=best_params["max_epochs"],
            pos_weight=best_params["pos_weight"],
            seed=SEED,
        )
        clf.fit(first_half_x.to_numpy(), first_half_y)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

    if profile_paths is not None:
        save_profile_metrics(task_name, criterion["name"], clf, second_half, second_half_x, second_half_y,
                             second_half_included, profile_paths, results_dir)
        return

    train_metrics = compute_metrics(first_half_y, clf.predict_proba(first_half_x)[:, 1])
    test_metrics = compute_metrics(second_half_y, clf.predict_proba(second_half_x)[:, 1])

    print(f"\n--- Base model metrics: {run_label} ---")
    print(f"{'Metric':<20}{'Train (1st half)':<20}{'Test (2nd half)':<20}")
    for key in METRIC_NAMES:
        print(f"{key:<20}{train_metrics[key]:<20.4f}{test_metrics[key]:<20.4f}")
    print()

    with open(f"{out_dir}/base_model_metrics.json", "w") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)

    # 6. MED3pa, with the Detectron (and FeatureResumer) applied in each profile
    datasets = DatasetsManager()
    datasets.set_from_data(dataset_type="training", observations=first_half_x.to_numpy(),
                           true_labels=first_half_y, column_labels=first_half_x.columns)
    datasets.set_from_data(dataset_type="testing", observations=second_half_x.to_numpy(),
                           true_labels=second_half_y, column_labels=second_half_x.columns)

    if criterion["name"] == "no_diabetes":
        # The shift feature is not an input of the model, so it cannot be summarized
        explainer = Explainer(methods=[Detectron])
    else:
        explainer = Explainer(methods=[Detectron, FeatureResumer(features=criterion["column"])])

    results = Med3paExperiment.run(
        datasets_manager=datasets,
        base_model_manager=BaseModelManager(model=clf),
        confidence_metric="sigmoidal_error",
        ipc_type="EnsembleRandomForestRegressor",
        ipc_params=IPC_PARAMS,
        apc_params=APC_PARAMS,
        ipc_grid_params=IPC_GRID,
        apc_grid_params=APC_GRID,
        # Only the profiles of minimal sample ratio 0 are analysed
        samples_ratio_min=0,
        samples_ratio_max=1,
        samples_ratio_step=5,
        evaluate_models=True,
        explainer=explainer,
    )

    # 7. Save results
    results.save(file_path=out_dir)
    visualize_mdr(result=results, filename=f"{out_dir}/mdr", show=False)
    visualize_tree(result=results, filename=f"{out_dir}/profiles_{datetime.now().strftime('%y%m%d%H%M%S')}",
                   open_results=False, scaler=scaler)

    metadata = {
        "task": task_name,
        "csv_path": csv_path,
        "exclusion_criterion": criterion,
        "seed": SEED,
        "target_col": TARGET_COL,
        "categorical_columns": categorical_columns,
        "n_rows_total": len(df),
        "n_first_half_before_exclusion": n_first_half_before,
        "n_first_half_after_exclusion": n_first_half_after,
        "n_second_half": len(second_half),
        "model_type": "TorchMLPClassifier",
        "mlp_best_cv_balanced_accuracy": study.best_value if study is not None else "",
        "mlp_best_params": {k: str(v) for k, v in best_params.items()} if best_params is not None else "",
    }
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_profile_metrics(task_name: str, criterion_name: str, clf, second_half: pd.DataFrame,
                         second_half_x: pd.DataFrame, second_half_y: np.ndarray, included: np.ndarray,
                         profile_paths: list, results_dir: str) -> None:
    """Base model metrics on each profile of the evaluation set, for all its
    patients and separately for those excluded from / included in the training
    population. Profiles are matched on raw (unscaled) feature values; profiles
    using features absent from this task are skipped."""
    subsets = {"all": np.ones(len(second_half), dtype=bool), "excluded": ~included, "included": included}

    all_metrics = {}
    for path in profile_paths:
        name = ", ".join(path)
        mask = profile_mask(second_half, parse_profile(path))
        if mask is None:
            continue
        all_metrics[name] = {}
        for subset, subset_mask in subsets.items():
            rows = mask & subset_mask
            proba = clf.predict_proba(second_half_x[rows].to_numpy())[:, 1] if rows.any() else np.array([])
            all_metrics[name][subset] = compute_metrics(second_half_y[rows], proba)

    out_dir = f"{results_dir}/profile_metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{task_name}_and_{criterion_name}_and_base_model_all_profiles_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved per-profile metrics to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile-metrics", metavar="CSV",
                        help="compute the base model metrics on the profiles listed in CSV "
                             "(written by experiments/extract_profiles.py) instead of running MED3pa")
    parser.add_argument("--test", action="store_true",
                        help="quick run on the first 200 rows of each dataset, saved in "
                             f"{RESULTS_DIR}_test")
    args = parser.parse_args()

    results_dir = f"{RESULTS_DIR}_test" if args.test else RESULTS_DIR
    n_rows = 200 if args.test else None
    profile_paths = load_profile_paths(args.profile_metrics) if args.profile_metrics else None

    for task_name, csv_path in TASKS.items():
        for criterion in EXCLUSION_CRITERIA:
            if criterion["name"] == "no_diabetes" and task_name not in DIABETES_TASKS:
                continue
            run_experiment(task_name, csv_path, criterion, results_dir, n_rows, profile_paths)


if __name__ == "__main__":
    main()
