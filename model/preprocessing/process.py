import ast
import logging
from typing import Any, Union

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from model.config.config import Config
from model.preprocessing.decode import count_elements, decode_input, normalize_abi
from model.preprocessing.knn import fill_storage_with_knn

logger = logging.getLogger(__name__)

def process_abi(
        contracts: pd.DataFrame
) -> pd.DataFrame:
    """
    Processes ABI-related columns in the contracts DataFrame.

    The function normalizes the `abi` and `proxy_abi` columns so that both are
    stored as valid Python lists. Invalid, missing or non-list ABI values are
    converted into empty lists.

    Args:
        contracts (pd.DataFrame): DataFrame containing contract metadata.

    Returns:
        pd.DataFrame: Copy of the contract DataFrame with normalized ABI columns.
    """
    contracts = contracts.copy()
    contracts["abi"] = contracts["abi"].apply(normalize_abi)
    contracts["proxy_abi"] = contracts["proxy_abi"].apply(normalize_abi)
    return contracts

def extract_signature_column(
        tsx: pd.DataFrame
) -> None:
    """
    Extracts the function signature from each transaction input.

    The function takes the first 10 characters of the transaction `input` field,
    corresponding to the Ethereum function selector, and stores it in a new
    `signature` column.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing the `input` column.

    Returns:
        None
    """
    tsx['signature'] = tsx['input'].map(lambda x: x[:10])

def filter_data_type(
        type: str
) -> bool:
    """
    Checks whether an ABI data type should be ignored during feature extraction.

    Args:
        type (str): ABI data type to check.

    Returns:
        bool: True if the data type should be filtered out, otherwise False.
    """
    filtered_types = {'address'}
    return type in filtered_types

def process_input(
        tsx: pd.DataFrame,
        contract: pd.Series,
        signature: str
) -> Union[pd.DataFrame, None]:
    """
    Decodes and processes transaction input data into numerical model features.

    The function decodes transaction inputs using the contract ABI, removes rows
    whose input cannot be decoded and creates additional feature columns from the
    decoded input parameters. Address parameters are ignored, while numerical,
    array-like and large integer values are transformed into model-compatible
    numerical features.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame for a specific contract and
            function signature.
        contract (pd.Series): Contract metadata containing ABI information.
        signature (str): Function signature currently being processed.

    Returns:
        pd.DataFrame: Transaction DataFrame enriched with input-derived feature
        columns. Returns an empty DataFrame if all inputs fail to decode.
    """
    tsx = tsx.copy()
    tsx['decode_input'] = tsx['input'].apply(decode_input, args=(contract,))#(type,name,value)
    if tsx['decode_input'].isna().all():
        logger.warning(f"SIGNATURE decode failure: contract={contract.name}, signature={signature}, rows={len(tsx)}")
        return pd.DataFrame() 
    tsx = tsx[tsx['decode_input'].notna()].copy() 
    new_cols=set() 
    try:
        #for index, row in tqdm(tsx.iterrows(), desc="process input"):
        for index, row in tsx.iterrows():
            
            tsx.at[index,'input_len'] = len(row['input'])
            new_cols.add('input_len')
            
            decode_input_value = row['decode_input']
            if not isinstance(decode_input_value, (list, tuple)): continue
            for (data_type, column_name, value) in decode_input_value:
                
                if(filter_data_type(data_type)): continue

                processed_value = process_type(value, data_type)

                if isinstance(processed_value, tuple):
                    tsx.at[index, data_type + '_'+ column_name] = processed_value[0]
                    tsx.at[index, column_name + '_iszero'] = processed_value[1]
                    new_cols.update([data_type + '_'+ column_name,column_name + '_iszero'])

                else:
                    tsx.at[index, data_type + '_'+ column_name] = processed_value
                    new_cols.add(data_type + '_'+ column_name)
    except Exception as e:
        logger.error(f"{signature}: {e}")

    new_cols = list(new_cols)
    if new_cols:
        tsx[new_cols] = tsx[new_cols].fillna(0)
        tsx[new_cols] = tsx[new_cols].infer_objects(copy=False)
    
    tsx.drop(columns=['decode_input'], inplace=True)

    return tsx

def process_type(
        value: str,
        type: str
) -> int:
    """
    Keywords uint8 to uint256 in steps of 8 (unsigned of 8 up to 256 bits) and int8 to int256. 
    uint and int are aliases for uint256 and int256, respectively.
    """
    INT_TYPES = [
        'uint', 'uint8', 'uint16', 'uint24', 'uint32', 'uint40', 'uint48', 'uint56', 'uint64', 
        'uint72', 'uint80', 'uint88', 'uint96', 'uint104', 'uint112', 'uint120', 'uint128', 
        'uint136', 'uint144', 'uint152', 'uint160', 'uint168', 'uint176', 'uint184', 'uint192', 
        'uint200', 'uint208', 'uint216', 'uint224', 'uint232', 'uint240', 'uint248', 'uint256', 
        'int', 'int8', 'int16', 'int24', 'int32', 'int40', 'int48', 'int56', 'int64', 'int72', 
        'int80', 'int88', 'int96', 'int104', 'int112', 'int120', 'int128', 'int136', 'int144', 
        'int152', 'int160', 'int168', 'int176', 'int184', 'int192', 'int200', 'int208', 'int216', 
        'int224', 'int232', 'int240', 'int248', 'int256'
    ]

    PROCESSING_INT_TYPES = ['uint8', 'uint16', 'uint24', 'uint32', 'uint40', 'uint48', 'uint56', 'uint64',
                            'int8', 'int16', 'int24', 'int32', 'int40', 'int48', 'int56', 'int64']

    UNPROCESSING_INT_TYPES = ['int', 'int72', 'int80', 'int88', 'int96', 'int104', 'int112', 'int120', 'int128',
                              'int136', 'int144', 'int152', 'int160', 'int168', 'int176', 'int184', 'int192',
                              'int200', 'int208', 'int216', 'int224', 'int232', 'int240', 'int248', 'int256',
                              'uint', 'uint72', 'uint80', 'uint88', 'uint96', 'uint104', 'uint112', 'uint120', 'uint128', 
                              'uint136', 'uint144', 'uint152', 'uint160', 'uint168', 'uint176', 'uint184', 'uint192', 
                              'uint200', 'uint208', 'uint216', 'uint224', 'uint232', 'uint240', 'uint248', 'uint256']
    
    value = str(value)

    if type in PROCESSING_INT_TYPES:
        return int(value)

    if type in UNPROCESSING_INT_TYPES:
        try:
            value_int = int(value)
            return (len(str(value)), 1 if value_int == 0 else 0)
        except ValueError:
            return len(str(value))

    if isinstance(value, list):
        return count_elements(value)
    else:
        return len(str(value))

def process_storage(
        tsx: pd.DataFrame,
        contract: pd.Series,
        signature: str,
        slotMap: dict,
) -> Union[tuple[pd.DataFrame, list[str]], None]:
    """
    Processes transaction storage data into numerical model features.

    The function parses the `storage_before` field, maps storage slots to stable
    numerical identifiers, converts storage values into model-compatible numeric
    representations and appends the resulting storage features to the transaction
    DataFrame. It also handles missing storage according to the configured storage
    management strategy and optionally fills missing values using a KNN model.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing storage information.
        contract (pd.Series): Contract metadata associated with the transactions.
        signature (str): Function signature currently being processed.
        slotMap (dict): Dictionary storing slot mappings, storage columns and
            optional KNN information for the current contract-signature pair.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - The transaction DataFrame enriched with storage-derived features.
            - The list of storage feature columns generated or used.
    """
    if tsx is None or tsx.empty or "storage_before" not in tsx.columns :return tsx, []
    
    if "slotMap" not in slotMap: slotMap["slotMap"] = {} #Siempre se actualiza
    if "knn" not in slotMap: slotMap["knn"] = None
    if "columns" not in slotMap: slotMap["columns"] = []
    slotMapping = slotMap["slotMap"]
    knnSlot = slotMap["knn"]

    tsx["storage_is_missing"] = tsx["storage_before"].isna().astype(int)
    tsx = process_unknown_strgs(tsx, knnSlot)
    storage_columns = ["strg_len","storage_is_missing"]    
    storage_rows = []
    nuevos_slots_storage = False
    for index, row in tsx.iterrows():
        new_values = {}
        new_values["strg_len"] = 0
        storage_raw = row.get("storage_before")
        try:
            if isinstance(storage_raw, dict):
                storage_data = storage_raw
            else:
                storage_data = ast.literal_eval(storage_raw)
            if not isinstance(storage_data, dict):
                storage_rows.append(new_values)
                continue
            new_values["strg_len"] = len(storage_data)
            i = 1
            for address, value in storage_data.items():
                col_slot = f"strgslot_{i}"
                col_val = f"strgval_{i}"
                col_big_val =f"strgbigval_{i}"

                pr_value_slot, new_slots= map_slot(address, slotMapping)
                pr_value_val,big_val = process_big_value(value)
                
                new_values[col_slot] = pr_value_slot
                new_values[col_val] = pr_value_val
                new_values[col_big_val] = 1 if big_val else 0

                nuevos_slots_storage = nuevos_slots_storage or new_slots

                if col_slot not in storage_columns:
                    storage_columns.append(col_slot)
                    storage_columns.append(col_val)
                    storage_columns.append(col_big_val)
                i += 1
        except Exception as e:
            logger.error(f"Error parsing storage in line {index}: {e}")
        storage_rows.append(new_values)
    
    storage_df = pd.DataFrame(storage_rows)
    
    storage_df.index = tsx.index
    tsx = pd.concat([tsx, storage_df], axis=1)
    #tsx = tsx.copy()
    if (Config.STRG_MANAGEMENT == 2):# Con KNN
        tsx = fill_storage_with_knn(tsx,knnSlot,storage_columns)
    tsx = tsx.infer_objects(copy=False)
    tsx = tsx.replace([np.inf, -np.inf], np.nan)
    tsx = tsx.fillna(0)
    
    tsx.drop(columns=["storage_before"], inplace=True, errors="ignore")
    tsx = tsx.reindex(sorted(tsx.columns), axis=1)

    if(nuevos_slots_storage):logger.warning("New storage info has been added. Consider retraining the KNN model if used.")
    
    return tsx, storage_columns

def map_slot(
        value: str,
        slotMap: dict
        )->tuple[int,bool]:
    """
    Maps a storage slot address to a stable numerical identifier.

    If the slot address already exists in the slot map, its existing identifier is
    returned. Otherwise, a new identifier is assigned and stored in the slot map.

    Args:
        value (str): Storage slot address.
        slotMap (dict): Dictionary mapping storage slot addresses to numerical
            identifiers.

    Returns:
        tuple[int, bool]: A tuple containing:
            - The numerical slot identifier.
            - True if a new slot was added, otherwise False.
    """
   
    if(value in slotMap): return slotMap[value], False
    else: 
        n = len(slotMap)
        slotMap[value] = n+1 
        return n+1, True

def process_big_value(value: str)->tuple[int,bool]:
    """
    Processes hexadecimal storage values into numerical representations.

    The function converts hexadecimal storage values into integers when they fit
    within a signed 64-bit range. Larger values are represented by their bit
    length to avoid excessively large numerical values.

    Args:
        value (str): Hexadecimal storage value.

    Returns:
        tuple[int, bool]: A tuple containing:
            - The processed numerical value.
            - True if the original value was too large and was represented by bit
              length, otherwise False.
    """
    try:
        if not isinstance(value, str) or not value.startswith("0x"):return 0
        n = int(value[2:], 16)
        if n == 0: return 0, False
        if n <= 2**63 - 1: return n,False
        else: return n.bit_length(),True
    except Exception:
        return 0, False

def process_unknown_strgs( 
        tsx: pd.DataFrame,
        modelKNN = None, 
    )-> pd.DataFrame:
    """
    Handles transactions with missing storage information.

    The function applies the storage management strategy defined in
    `Config.STRG_MANAGEMENT`. Depending on the selected mode, rows without storage
    may be removed, filled with empty dictionaries, or prepared for KNN-based
    storage imputation.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing the `storage_before`
            column.
        modelKNN: Optional trained KNN model used when KNN-based storage handling
            is enabled.

    Returns:
        pd.DataFrame: Transaction DataFrame after applying the selected missing
        storage handling strategy.
    """
    if (Config.STRG_MANAGEMENT == 0):#Op1: No usar los datos que no tienen info de storage
        tsx = tsx.dropna(subset=['storage_before']).copy()
        tsx.drop(columns=["storage_is_missing"], inplace=True, errors="ignore")
    elif (Config.STRG_MANAGEMENT == 1): #Op2: Rellenar con diccionarios vacios
        tsx['storage_before'] = tsx['storage_before'].fillna("{}").astype(str)
    elif (Config.STRG_MANAGEMENT == 2): #Op3: KNN
        if modelKNN is None:
            tsx = tsx.dropna(subset=["storage_before"]).copy()
            logger.warning(f"[KNN] Couln't find avaliable KNN model. "f" Lines without storage info will be deleted from the dataframe.")
        else:
            tsx["storage_before"] = tsx["storage_before"].fillna("{}").astype(str)
    else:
        raise ValueError(f"Invalid STRG_MANAGEMENT: {Config.STRG_MANAGEMENT}")
    return tsx



