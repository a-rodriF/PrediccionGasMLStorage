import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

from model.modeling.MyModel import MyModel

GLOBAL_N_JOBS = 1

def train(
        signature_tsx: pd.DataFrame,
        strg_columns: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Trains gas prediction models for a specific contract-function signature dataset.

    The function prepares the feature matrix and target variable, validates that
    all model inputs are numeric, sanitizes feature names, splits the data into
    training and test sets, scales the features and trains two groups of models:
    one using storage-related features and another excluding them.

    Args:
        signature_tsx (pd.DataFrame): Transaction DataFrame filtered for a
            specific contract address and function signature.
        strg_columns (list[str] | None, optional): List of storage-related
            feature columns to identify and remove for the no-storage training
            scenario. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing the trained models, scaler,
        sampled test data, dataset size, storage feature list and model names.
    """
    models = dict()

    signature_tsx = signature_tsx.drop(['block_timestamp'], axis=1)

    base_drop = ['hash','to_address', 'input', 'signature', 'receipt_gas_used']#igual que antes pero añades hash
    strg_columns = strg_columns or []
    strg_columns = [c for c in strg_columns if c in signature_tsx.columns]

    X = signature_tsx.drop(columns=base_drop, errors="ignore")
    y = signature_tsx['receipt_gas_used'] 

    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print("Columnas object:", obj_cols)
        for col in obj_cols:
            print(col, X[col].head(3).tolist())
        raise ValueError(f"Hay columnas no numéricas: {obj_cols}")
   
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in X.columns]
    strg_columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(col)) for col in strg_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    random_index = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    models['X_test'] = X_test.iloc[random_index] #ejemplo
    models['y_test'] = y_test.iloc[random_index] #ejemplo
    models['size'] = len(y)
    models['storage info'] =strg_columns

    scaler = RobustScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    models['scaler'] = scaler

    models['with storage'],m = train_model_types(X_train, X_test,y_train,y_test)
    
    X_train = X_train.drop(columns=strg_columns, errors="ignore")
    X_test = X_test.drop(columns=strg_columns, errors="ignore")
    models['no storage'],_ = train_model_types(X_train, X_test,y_train,y_test)
    
    models['modelos_usados'] = m
    return models

def train_model_types(
        X_train,
        X_test,
        y_train,
        y_test,
)->tuple[Dict[str,Any],list[str]]:
    """
    Trains multiple regression model types and stores their evaluation results.

    The function initializes several regression algorithms, wraps each one in the
    custom `MyModel` class, trains it using the provided training and test data
    and returns the trained model objects.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target values.
        y_test: Test target values.

    Returns:
        tuple[Dict[str, Any], list[str]]: A tuple containing:
            - A dictionary with trained `MyModel` instances indexed by model name.
            - A list-like object with the names of the trained model types.
    """
    modelos ={'LinearRegression': LinearRegression(),
              'SVRlinear': LinearSVR(dual=False, loss='squared_epsilon_insensitive', C=1.0, epsilon=0.1, random_state=42, max_iter=10000),
              'GradientBoost':GradientBoostingRegressor(),
              'Ridge':Ridge(alpha=10.0, solver="lsqr"),
              'RandomForest': RandomForestRegressor(n_estimators=100,n_jobs=GLOBAL_N_JOBS),
              'XGBoost': XGBRegressor(n_jobs = GLOBAL_N_JOBS),
              'SVRrbf': SVR(kernel="rbf"),#Muy malo en set reducido
              }
    models = dict()
    for clave, valor in modelos.items():
        m = MyModel(valor) 
        m.train_model(X_train, X_test, y_train, y_test)
        models[clave] = m

    return models, modelos.keys()

