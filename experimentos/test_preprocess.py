import json
import pandas as pd
from model.data_processing.reader import merge, storage_info_available
from model.preprocessing.process import process_abi, process_input, extract_signature_column, process_storage
from model.data_processing.reader import read, read_all, read_merged

MIN_LEN_TSX = 100

DATA_PATH = 'data/' 

def main():

    print("[1/4] Leyendo datos...")
   
    #tsx, contracts = read_all()
    tsx, contracts = read()
    #tsx, contracts = read_merged()

    print("Contracts columns:", contracts.columns.tolist())
    print("TSX columns:", tsx.columns.tolist())

    print("[2/4] Processing ABI...")
    contracts = process_abi(contracts)

    #contracts = contracts.set_index("contract")

    print("Contracts index OK")

    print("[3/4] Searching for valid contract and signatures...")

    tsx_grouped = tsx.groupby("to_address")

    selected_contract = None
    selected_contract_info = None
    selected_signature = None
    selected_signature_tsx = None

    for contract_address, contract_tsx in tsx_grouped:
        if contract_address not in contracts.index:continue
        contract_tsx = contract_tsx.copy()
        
        has_storage = contract_tsx['storage_before'].dropna().apply(
            lambda x: x != "{}" and x != ""
        ).any()
        if not has_storage:continue
        
        extract_signature_column(contract_tsx)
        signature_grouped = contract_tsx.groupby("signature")
        for signature, signature_tsx in signature_grouped:
            if len(signature_tsx) > MIN_LEN_TSX:
                selected_contract = contract_address
                selected_contract_info = contracts.loc[contract_address]
                selected_signature = signature
                selected_signature_tsx = signature_tsx.copy()
                break

        if selected_contract is not None:
            break

    if selected_contract is None:
        raise ValueError("Couldn't find a contract with enough transactions.")

    print("[4/4] Processing a single signature...")

    print("Contract:", selected_contract)
    print("Signature:", selected_signature)
    print("N tx:", len(selected_signature_tsx))

    processed_df = process_input(
        selected_signature_tsx,
        selected_contract_info,
        selected_signature
    )
    processed_df,storage_columns = process_storage(
        processed_df,
        selected_contract_info,
        selected_signature
    ) 

    print("\n=== PROCESSED DATAFRAME ===")

    print("Shape:", processed_df.shape)

    print("\n=== COLUMNS ===")
    print(processed_df.columns.tolist())

    print("\n=== HEAD ===")
    print(processed_df.head())

    if "receipt_gas_used" in processed_df.columns:

        print("\n=== TARGET ===")
        print(processed_df["receipt_gas_used"].head())

        X = processed_df.drop(
            columns=["hash", "to_address", "input", "signature", "receipt_gas_used"],
            errors="ignore"
        )

        print("\n=== FEATURES INPUTED IN THE MODEL ===")
        print(X.columns.tolist())

        print("\n=== HEAD FEATURES ===")
        print(X.head())

        #INFO
        processed_df.to_csv('prueba.csv', index=False, sep=';', encoding='utf-8-sig')
        storage_info_available(tsx)


if __name__ == "__main__":
    main()