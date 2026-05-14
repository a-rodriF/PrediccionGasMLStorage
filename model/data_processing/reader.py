import json
import os
import pickle
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from model.config.config import Config
from model.data_processing.storage import extract_storage_byHash, normalize_hash
from model.export.dumper import save_dataFrame
from model.preprocessing.process import extract_signature_column


def merge(tsx, strg) -> pd.DataFrame:
    """
    Merges transaction data with storage information using the transaction hash.

    The function normalizes the hash values in both datasets and performs a left
    join so that all transactions are preserved, even when no matching storage
    information is available.

    Args:
        tsx (pd.DataFrame): DataFrame containing transaction information.
        strg (pd.DataFrame): DataFrame containing storage information, including
            the transaction hash and the storage state before execution.

    Returns:
        pd.DataFrame: Transaction DataFrame enriched with the `storage_before`
        column when matching storage information is found.
    """
    if (Config.PRE_MERGE_INFO):pre_merge_info(tsx,strg)
    #tsx = tsx.copy()
    #strg = strg.copy()
    tsx['hash'] = tsx['hash'].apply(normalize_hash)
    strg['hash'] = strg['hash'].apply(normalize_hash)

    tsx = tsx.merge(
        strg[['hash', 'storage_before']],
        on='hash',
        how='left'
    )
    return tsx

def read() -> Tuple[pd.DataFrame, pd.DataFrame,dict]:
    """
    Reads the sample transaction, contract and storage datasets.

    This function loads the contract metadata, the slot map, the sample
    transactions and the sample storage JSON file. It then merges transaction
    data with storage information and normalizes contract addresses.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: A tuple containing:
            - The transaction DataFrame enriched with storage information.
            - The contract DataFrame indexed by normalized contract address.
            - The slot map dictionary.
    """
    contracts = pd.read_csv(Config.DATA_PATH + 'contracts100.csv', sep=';',index_col='contract') 
    contracts.index = contracts.index.map(normalize_hash)
    slotMap = read_SlotMap()
    
    tsx = pd.read_csv(Config.DATA_PATH + 'example_tsx.csv', sep=';')
    
    with open(Config.DATA_PATH + 'example_strg.json', 'r', encoding='utf-8') as f:
        storage_json = json.load(f)
    strg = pd.DataFrame(storage_json)
    
    tsx = merge(tsx,strg)
    tsx["to_address"] = tsx["to_address"].apply(normalize_hash)
    return tsx,  contracts, slotMap

def read_all() -> Tuple[pd.DataFrame, pd.DataFrame,dict]:
    """
    Reads all transaction files, extracts the corresponding storage information
    and merges both sources into a single dataset.

    The function loads all transaction CSV files from the configured transaction
    directory, extracts storage information for the detected transaction hashes,
    merges both datasets, optionally saves missing-storage hashes, displays
    storage coverage information and stores the merged dataset if configured.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: A tuple containing:
            - The complete transaction DataFrame enriched with storage information.
            - The contract DataFrame indexed by normalized contract address.
            - The slot map dictionary.
    """
    #contratos
    contracts = pd.read_csv(Config.DATA_PATH+'contracts100.csv', sep=';', index_col='contract')
    contracts.index = contracts.index.map(normalize_hash)
    #slotMap
    slotMap = read_SlotMap()
    
    #información general transacciones
    df_tsx = []
    for filename in tqdm(os.listdir(Config.TSX_PATH), desc="Reading Transactions"):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(Config.TSX_PATH, filename)
        df = pd.read_csv(filepath, sep=';')#, index_col='hash'
        df_tsx.append(df)
    tsx = pd.concat(df_tsx)

    hashes_buscados = set(tsx['hash'].apply(normalize_hash).unique())
    
    strg = extract_storage_byHash(
        strg_path=Config.STRG_PATH,
        hashes_buscados=hashes_buscados,
        n_processes=Config.N_JSON_PROCESES
    )
    print("Storage hashes found") 

    tsx = merge(tsx,strg)#merge tsx info with strg info according to the hash
    print("Merge is complete")
    if(Config.SAVE_HASH_MISSING_INFO):
        save_tsx_without_match_with_date(
            tsx,
            'tsx_without_storage.csv'
        )   
    if (Config.SHOW_STRG_INFO):storage_info_available(tsx)
    tsx["to_address"] = tsx["to_address"].apply(normalize_hash)
    if Config.ONLY_MERGE: save_dataFrame(tsx,Config.MRGD_PATH)
    return tsx, contracts, slotMap
    
def read_merged()->Tuple[pd.DataFrame, pd.DataFrame,dict]:
    """
    Reads previously merged transaction datasets from disk.

    The function loads contract metadata, the slot map and all merged CSV files
    from the configured merged-data directory. It normalizes contract addresses
    and, if configured, reduces the dataset by removing contract-signature groups
    with insufficient transaction samples.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: A tuple containing:
            - The merged transaction DataFrame.
            - The contract DataFrame indexed by normalized contract address.
            - The slot map dictionary.
    """
    contracts = pd.read_csv(Config.DATA_PATH+'contracts100.csv', sep=';', index_col='contract')
    contracts.index = contracts.index.map(normalize_hash)
    slotMap = read_SlotMap()
    
    df_merged = []
    for filename in tqdm(os.listdir(Config.MRGD_PATH), desc="Reading Transactions"):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(Config.MRGD_PATH, filename)
        df = pd.read_csv(filepath, sep=';')#, index_col='hash'
        df_merged.append(df)
    tsx = pd.concat(df_merged)
    tsx["to_address"] = tsx["to_address"].apply(normalize_hash)
    
    if Config.ONLY_REDUCE:
        tsx = reduce(tsx)
        if (Config.SHOW_STRG_INFO):storage_info_available(tsx)
        save_dataFrame(tsx,Config.RDC_PATH)
    return tsx, contracts, slotMap

def read_reduced()->Tuple[pd.DataFrame, pd.DataFrame,dict]:
    """
    Reads the reduced transaction dataset from disk.

    The function loads contract metadata, the slot map and all reduced CSV files
    from the configured reduced-data directory. It also normalizes contract
    addresses before returning the data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: A tuple containing:
            - The reduced transaction DataFrame.
            - The contract DataFrame indexed by normalized contract address.
            - The slot map dictionary.
    """
    contracts = pd.read_csv(Config.DATA_PATH+'contracts100.csv', sep=';', index_col='contract')
    contracts.index = contracts.index.map(normalize_hash)
    slotMap = read_SlotMap()

    df_merged = []
    for filename in tqdm(os.listdir(Config.RDC_PATH), desc="Reading Transactions"):
    
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(Config.RDC_PATH, filename)
        df = pd.read_csv(filepath, sep=';')#, index_col='hash'
        df_merged.append(df)
    tsx = pd.concat(df_merged)
    tsx["to_address"] = tsx["to_address"].apply(normalize_hash)
    
    return tsx, contracts, slotMap

def read_SlotMap()-> dict:
    """
    Loads and combines slot mapping information and trained KNN storage models.

    The function reads the slot map JSON file and the serialized KNN models if
    they exist. It then combines both sources into a unified dictionary organized
    by contract address and function signature.

    Returns:
        dict: A nested dictionary containing, for each contract and signature,
        the storage columns, slot map, metrics and KNN model information.
    """
    if os.path.exists(Config.SLOT_PATH):
        with open(Config.SLOT_PATH, 'r', encoding='utf-8') as f:
            mapa = json.load(f)
    else:
        mapa = {}
    
    if os.path.exists(Config.KNN_PATH):
        with open(Config.KNN_PATH, "rb") as f:
            knn_models = pickle.load(f)
    else:
        knn_models = {}
    #Unir mapas por signature
    slotMap = {}
    all_contracts = set(mapa.keys()) | set(knn_models.keys())
    for contract in all_contracts:
        slotMap[contract] = {}
        json_signatures = mapa.get(contract, {})
        knn_signatures = knn_models.get(contract, {})
        all_signatures = set(json_signatures.keys()) | set(knn_signatures.keys())
        for signature in all_signatures:
            json_data = json_signatures.get(signature, {})
            knn_data = knn_signatures.get(signature, {})
            knn_columns = mapa.get("columns", [])
            json_columns = json_data.get("columns", [])
            columns = knn_columns if knn_columns else json_columns
            slotMap[contract][signature] = {
                "columns": columns,
                "slotMap": json_data.get("slotMap", {}),
                "metrics": json_data.get("metrics", None),
                "knn": knn_data
            }
    return slotMap

def storage_info_available(tsx):
    """
    Displays storage availability statistics for the transaction dataset.

    The function calculates the total number of transactions, the number of
    transactions with available storage information, the number without storage
    information and the overall storage coverage percentage.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing the
            `storage_before` column.
    """
    #tsx = tsx.copy()
    total = len(tsx)
    with_storage = tsx['storage_before'].notna().sum()
    coverage = with_storage / total if total > 0 else 0

    print(f"Total transacciones: {total}")
    print(f"Con storage: {with_storage}")
    print(f"Sin storage: {total - with_storage}")
    print(f"Cobertura: {coverage:.2%}")

def pre_merge_info(tsx,strg):
    """
    Displays diagnostic information before merging transactions and storage data.

    The function prints the number of unique transaction hashes, the number of
    unique storage hashes, the available storage columns, the total number of
    storage rows and the number of non-null `storage_before` entries.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing transaction hashes.
        strg (pd.DataFrame): Storage DataFrame containing hashes and storage
            information.
    """
    print("Number of Transaction hashes:", tsx['hash'].apply(normalize_hash).nunique())
    print("Number of Storage hashes:", strg['hash'].apply(normalize_hash).nunique())
    print("Storage columns:", strg.columns.tolist())
    print("Number of Storage rows:", len(strg))
    print("Non-null storage_before rows in storage:", strg['storage_before'].notna().sum() if 'storage_before' in strg.columns else 'no existe')
    
def save_tsx_without_match_with_date(tsx: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Saves transaction hashes without matching storage information to a CSV file.

    The function identifies transactions whose `storage_before` value is missing,
    keeps their hash and block timestamp, removes duplicated hashes and exports
    the result to the specified CSV file.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame after the merge with storage
            information.
        output_file (str): Path of the CSV file where unmatched transaction
            hashes will be saved.

    Returns:
        pd.DataFrame: DataFrame containing the unmatched transaction hashes and
        their corresponding block timestamps.
    """
    tsx_aux = tsx.copy()
    tsx_aux['hash'] = tsx_aux['hash'].apply(normalize_hash)

    tsx_no_match = tsx_aux.loc[
        tsx_aux['storage_before'].isna(),
        ['hash', 'block_timestamp']
    ].drop_duplicates(subset='hash')

    tsx_no_match.to_csv(output_file, index=False)

    print(f"Transaction hashes without storage info saved to: {output_file}")
    print(f"Number of hashes without storage: {len(tsx_no_match)}")

    return tsx_no_match

def reduce(tsx:pd.DataFrame):
    """
    Reduces the transaction dataset by keeping only contract-signature groups
    with enough samples.

    The function extracts the function signature from each transaction input and
    filters out groups whose number of transactions is below the configured
    minimum threshold.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing contract addresses
            and input data.

    Returns:
        pd.DataFrame: Filtered transaction DataFrame containing only groups with
        at least `Config.MIN_LEN_TSX` transactions.
    """
    extract_signature_column(tsx)
    tsx_filtrado = tsx.groupby(['to_address', 'signature']).filter(
        lambda x: len(x) >= Config.MIN_LEN_TSX
    )
    return tsx_filtrado
