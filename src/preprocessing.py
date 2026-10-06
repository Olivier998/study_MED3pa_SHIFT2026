"""Dataset cleaning, categorical encoding and exclusion criteria."""

import operator as op
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

OPS = {
    "<": op.lt,
    "<=": op.le,
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
}


def clean_dataset(df: pd.DataFrame, target_col: str, max_missing_ratio: float = 0.20) -> pd.DataFrame:
    """Drop identifier/indicator columns and columns with too many missing values,
    and cast a "True"/"False" string target to 0/1."""
    cols_to_drop = [c for c in df.columns if "stay_id" in c or c.endswith("_present")]
    if "Unnamed: 0" in df.columns:
        cols_to_drop.append("Unnamed: 0")
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"True": 1, "False": 0}).astype(int)

    missing_ratio = df.isna().mean()
    too_sparse = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if too_sparse:
        print(f"Removing columns with >{max_missing_ratio:.0%} missing values: {too_sparse}")
        df = df.drop(columns=too_sparse)

    return df


def infer_categorical_columns(df: pd.DataFrame, target_col: str) -> list:
    """Non-numeric columns (object, category, bool, string), excluding the target."""
    return [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]


def fit_categorical_encoder(df: pd.DataFrame, categorical_columns: list) -> Optional[OneHotEncoder]:
    """Fit a one-hot encoder on the categorical columns of the full dataset, so that
    both halves of the split share the same encoding."""
    if not categorical_columns:
        return None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[categorical_columns])
    return encoder


def encode_categoricals(df: pd.DataFrame, categorical_columns: list,
                        encoder: Optional[OneHotEncoder]) -> pd.DataFrame:
    """Replace the categorical columns by their one-hot encoding (appended at the end)."""
    if encoder is None:
        return df.copy()
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index,
    )
    return pd.concat([df.drop(columns=categorical_columns), encoded], axis=1)


def criterion_mask(df: pd.DataFrame, criterion: dict) -> pd.Series:
    """Rows satisfying a {"column", "operator", "value"} criterion."""
    return OPS[criterion["operator"]](df[criterion["column"]], criterion["value"])


def apply_criterion(df: pd.DataFrame, criterion: dict) -> pd.DataFrame:
    """Keep only the rows satisfying the criterion."""
    return df[criterion_mask(df, criterion)].reset_index(drop=True)


def profile_mask(df: pd.DataFrame, conditions: list) -> Optional[np.ndarray]:
    """Rows of `df` belonging to a profile, given as a list of criteria.

    Column names are matched after replacing whitespace with underscores, as in
    the MED3pa profile trees. Returns None if the profile uses a column absent
    from `df` (i.e. the profile comes from another task).
    """
    columns = ["_".join(c.strip().split()) for c in df.columns]
    mask = np.ones(len(df), dtype=bool)
    for cond in conditions:
        if cond["column"] not in columns:
            return None
        values = df.iloc[:, columns.index(cond["column"])].to_numpy()
        mask &= OPS[cond["operator"]](values, cond["value"])
    return mask
