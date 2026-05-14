from model.export.dumper import import_models
from pprint import pprint

models = import_models("data/models.pkl")

for contract, signatures in models.items():
    if isinstance(signatures, dict) and len(signatures) > 0:
        print("CONTRACT:", contract)
        print("SIGNATURES:", signatures.keys())

        first_sig = next(iter(signatures))
        print("FIRST SIGNATURE:", first_sig)
        print("TYPE:", type(signatures[first_sig]))
        pprint(signatures[first_sig])
        break