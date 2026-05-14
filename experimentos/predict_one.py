from model.export.dumper import import_models

models = import_models("data/models.pkl")

contract = "0x00000000000000adc04c56bf30ac9d3c0aaf14dc"
signature = "0x00000000"

payload = models[contract][signature]

lr_wrapper = payload["LinearRegression"]
gb_wrapper = payload["GradientBoost"]

X_test = payload["X_test"].copy()
y_test = payload["y_test"]

if "block_timestamp" in X_test.columns:
    X_eval = X_test.drop(columns=["block_timestamp"])
else:
    X_eval = X_test

x_one = X_eval.iloc[[0]]   
y_real = y_test.iloc[0]

lr_model = getattr(lr_wrapper, "model", lr_wrapper)
gb_model = getattr(gb_wrapper, "model", gb_wrapper)

lr_pred = lr_model.predict(x_one)[0]
gb_pred = gb_model.predict(x_one)[0]

print("Contract:", contract)
print("Signature:", signature)
print("Hash:", X_eval.index[0])
print("Real Gas:", y_real)
print("LinearRegression prediction:", lr_pred)
print("GradientBoost prediction:", gb_pred)