from model.export.dumper import import_models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

models = import_models("data/models.pkl")

rows = []

for contract, signatures in models.items():
    if not isinstance(signatures, dict) or len(signatures) == 0:
        continue

    for signature, payload in signatures.items():
        if not isinstance(payload, dict):
            continue

        lr = payload.get("LinearRegression")
        gb = payload.get("GradientBoost")
        X_test = payload.get("X_test")
        y_test = payload.get("y_test")
        size = payload.get("size")

        if X_test is None or y_test is None:
            continue

        X_eval = X_test.copy()
        if "block_timestamp" in X_eval.columns:
            X_eval = X_eval.drop(columns=["block_timestamp"])

        for model_name, model_obj in [
            ("LinearRegression", lr),
            ("GradientBoost", gb),
        ]:
            if model_obj is None:continue

            estimator = getattr(model_obj, "model", model_obj)

            try:
                y_pred = estimator.predict(X_eval)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                rows.append({
                    "contract": contract,
                    "signature": signature,
                    "model": model_name,
                    "size": size,
                    "n_test": len(y_test),
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mape": mape,
                })
            except Exception as e:
                rows.append({
                    "contract": contract,
                    "signature": signature,
                    "model": model_name,
                    "size": size,
                    "n_test": len(y_test),
                    "mse": None,
                    "mae": None,
                    "r2": None,
                    "mape": None,
                    "error": str(e),
                })

results = pd.DataFrame(rows)

print(results.head(20))
print("\nModel summary:")
print(results.groupby("model")[["mse", "mae", "r2", "mape"]].mean())

results.to_csv("data/evaluation_results.csv", index=False)
print("\nSaved in data/evaluation_results.csv")