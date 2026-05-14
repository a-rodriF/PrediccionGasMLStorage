import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from model.config.config import Config

logger = logging.getLogger(__name__)
tolerance = 0.05  

def train_knn_storage(
    df: pd.DataFrame,
    storage_cols: list[str],
    n_neighbors: int = 5,
    limitKNN: int = Config.MAX_LEN_KNN,):
    """
    Trains a KNN model to predict missing storage feature values.

    The function prepares a dataset using non-storage numerical features as
    predictors and storage columns as multi-output targets. It optionally reduces
    the training size through stratified sampling by gas usage, trains a
    distance-weighted KNN model and computes evaluation metrics on a test split.

    Args:
        df (pd.DataFrame): Transaction DataFrame containing input, gas and
            storage-related features.
        storage_cols (list[str]): List of storage columns to be predicted.
        n_neighbors (int, optional): Number of neighbors used by the KNN model.
            Defaults to 5.
        limitKNN (int, optional): Maximum number of samples used to train the KNN
            model. Defaults to `Config.MAX_LEN_KNN`.

    Returns:
        dict | None: Dictionary containing the trained KNN pipeline and its
        evaluation metrics, or None if there is not enough valid data to train
        the model.
    """
    df = df.copy()
    df = stratified_sample_by_gas(df, limitKNN) 
    storage_cols = storage_cols.copy()
    drop_cols = ["hash","to_address","input", "signature","block_timestamp","receipt_gas_used", 
                 "storage_before","storage_is_missing",]
    storage_cols = [c for c in storage_cols if c in df.columns and c != "storage_is_missing"]
    if not storage_cols:
        logger.error("No valid storage columns/data available to train KNN.")
        return None

    X = df.drop(columns=drop_cols + storage_cols, errors="ignore")
    y = df[storage_cols]

    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in X.columns]
    y.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in y.columns]

    X = X.select_dtypes(include=[np.number])
    y = y.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) < Config.MIN_LEN_KNN:
        logger.info(f"[SKIPPED KNN] Too few samples for KNN: {len(X)} samples.")
        return None
    if X.empty or y.empty:
        logger.error("X or y are empty for KNN.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    k = min(n_neighbors, len(X_train))
    model = Pipeline([
        ("scaler", RobustScaler()),
        ("knn", MultiOutputRegressor(
            KNeighborsRegressor(
                n_neighbors=k,
                weights="distance",
                n_jobs=1
            )
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cálculo de métricas
    exact_match = np.all(np.rint(y_pred) == np.rint(y_test.values), axis=1).mean()
    denom = np.where(y_test.values == 0, 1, np.abs(y_test.values))
    relative_error = np.abs(y_pred - y_test.values) / denom
    hits_per_col = (relative_error <= tolerance).mean(axis=0)
    #mae_per_col = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    #r2_per_col = r2_score(y_test, y_pred, multioutput='raw_values')
    
    metrics = {
        "global_mae": mean_absolute_error(y_test, y_pred),
        "global_r2": r2_score(y_test, y_pred),
        "exact_match_ratio": exact_match, # % de filas donde acertó TODO perfecto
        "reliability_score": hits_per_col.mean(), # % de aciertos promedio dentro del margen
        #"per_column_reliability": dict(zip(storage_cols, hits_per_col)),
        "samples_used": len(X)
    }
    model.fit(X, y)

    return {
        "model": model,
        "metrics": metrics
    }

def fill_storage_with_knn(
        tsx: pd.DataFrame,
        modelKNN,
        storage_columns: list[str],
) -> pd.DataFrame:
    """
    Fills missing storage feature values using a trained KNN model.

    The function identifies rows marked as missing storage, builds the prediction
    feature matrix, aligns it with the features used during KNN training and
    replaces missing storage values with rounded non-negative predictions.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing storage features and
            a `storage_is_missing` indicator column.
        modelKNN: Trained KNN model or dictionary containing the model under the
            `model` key.
        storage_columns (list[str]): List of storage-related columns to fill.

    Returns:
        pd.DataFrame: Copy of the transaction DataFrame with predicted storage
        values filled where storage information was missing.
    """
    tsx = tsx.copy()
    if modelKNN is None: return tsx
    if isinstance(modelKNN, dict):
        model = modelKNN.get("model")
    else:
        model = modelKNN
    if model is None: return tsx
    storage_columns = [c for c in storage_columns if c in tsx.columns]
    storage_columns.remove("storage_is_missing")
    if not storage_columns:return tsx

    pred_mask = tsx["storage_is_missing"] == 1
    if not pred_mask.any(): return tsx
    drop_cols = ["hash","to_address","input","signature","block_timestamp","receipt_gas_used", "storage_before",]

    X_pred = tsx.loc[pred_mask].drop(
        columns=drop_cols + storage_columns,
        errors="ignore"
    )
    X_pred = X_pred.select_dtypes(include=[np.number])
    if X_pred.empty: return tsx
    
    X_pred.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in X_pred.columns]
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    y_pred = model.predict(X_pred)

    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.rint(y_pred)
    tsx.loc[pred_mask, storage_columns] = y_pred
    return tsx

def stratified_sample_by_gas(df, max_size=5000, bins=10):
    """
    Performs stratified sampling based on gas usage.

    The function reduces the dataset size while preserving the distribution of
    `receipt_gas_used`. It divides gas usage into quantile-based bins and samples
    proportionally from each bin.

    Args:
        df (pd.DataFrame): Input DataFrame containing the `receipt_gas_used`
            column.
        max_size (int, optional): Maximum number of rows to keep. Defaults to
            5000.
        bins (int, optional): Number of quantile bins used for stratification.
            Defaults to 10.

    Returns:
        pd.DataFrame: Stratified sample of the original DataFrame. If the input
        DataFrame is already smaller than or equal to `max_size`, it is returned
        unchanged.
    """
    if len(df) <= max_size:
        return df
    df = df.copy()
    
    df["gas_bin"] = pd.qcut(
        df["receipt_gas_used"],
        q=bins,
        duplicates="drop"
    )
   
    sampled = df.groupby("gas_bin", group_keys=False, observed=False).apply(
        lambda x: x.sample(
            max(1, int(len(x) * max_size / len(df))),
            random_state=42
        )
    )
    return sampled.drop(columns=["gas_bin"])

