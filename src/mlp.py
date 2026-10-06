"""PyTorch MLP base model and its Optuna hyperparameter search."""

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset


class TorchMLP(nn.Module):
    """Fully connected network with one logit output."""

    def __init__(self, input_dim, hidden_layers, activation="relu", dropout=0.2):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchMLPClassifier:
    """Binary classifier with a scikit-learn-like interface (fit / predict / predict_proba).

    `fit` accepts per-sample weights, which the Detectron uses to train its
    constrained disagreement classifiers from copies of this model.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 64),
        activation="relu",
        lr=1e-3,
        batch_size=64,
        max_epochs=80,
        pos_weight=1.0,
        dropout=0,
        seed=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.pos_weight = float(pos_weight)
        self.dropout = dropout
        self.seed = seed
        self.model = None

    def _weighted_bce_with_logits(self, logits, targets):
        """Per-sample BCE-with-logits (no reduction), with `pos_weight` applied
        to positive samples, as in torch.nn.BCEWithLogitsLoss(pos_weight)."""
        # Numerically stable form: max(z, 0) - z*y + log(1 + exp(-|z|))
        max_val = torch.clamp(logits, min=0)
        loss = max_val - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))

        pw = torch.tensor(self.pos_weight, dtype=torch.float32, device=logits.device)
        return torch.where(targets == 1, loss * pw, loss)

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        torch.manual_seed(self.seed)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if sample_weight is None:
            w = torch.ones_like(y)
        else:
            w = torch.tensor(sample_weight, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X, y, w), batch_size=self.batch_size, shuffle=True)

        self.model = TorchMLP(X.shape[1], self.hidden_layer_sizes, self.activation, dropout=self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.max_epochs):
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                loss = (self._weighted_bce_with_logits(self.model(xb), yb) * wb).mean()
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X)).numpy()
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tune_mlp(X: pd.DataFrame, y, n_trials: int, seed: int, max_epochs: int = 80):
    """Select MLP hyperparameters by maximizing the 5-fold stratified CV balanced
    accuracy with Optuna (TPE sampler).

    Returns the selected parameters and the Optuna study.

    Note: dropout is part of the search space but is not included in the returned
    parameters, so the final model is trained without dropout (the
    TorchMLPClassifier default). This matches the setup used for the paper.
    """
    X_arr = X.to_numpy()
    y_arr = np.asarray(y)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_arr, y_arr))

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 8, 128, log=True) for i in range(n_layers)
        )
        pos_weight = trial.suggest_float("pos_weight", 0.5, 25, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)

        scores = []
        for train_idx, val_idx in folds:
            clf = TorchMLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                lr=lr,
                max_epochs=max_epochs,
                pos_weight=pos_weight,
                dropout=dropout,
                seed=seed,
            )
            clf.fit(X_arr[train_idx], y_arr[train_idx])
            scores.append(balanced_accuracy_score(y_arr[val_idx], clf.predict(X_arr[val_idx])))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_params = {
        "hidden_layer_sizes": tuple(best[f"n_units_l{i}"] for i in range(best["n_layers"])),
        "activation": best["activation"],
        "learning_rate_init": best["learning_rate_init"],
        "pos_weight": best["pos_weight"],
        "max_epochs": max_epochs,
        "seed": seed,
    }
    return best_params, study
