import logging
from multiprocessing import Process, Queue, cpu_count
from queue import Empty

import pandas as pd
from tqdm import tqdm

from model.config.config import Config
from model.modeling.trainer import train
from model.preprocessing.knn import train_knn_storage
from model.preprocessing.process import process_input,process_storage, extract_signature_column

logger = logging.getLogger(__name__)

MAX_PROCESSES = max(cpu_count() // 2, 1)

def process(
        tsx: pd.DataFrame,
        contracts: pd.DataFrame,
        slotMap: dict,
        models: dict,
) -> None:
    """
    Processes the complete transaction dataset grouped by contract address.

    The function groups transactions by destination contract address, checks
    whether each contract exists in the contract metadata, initializes the
    corresponding model and slot map entries, and processes each contract
    independently.

    Args:
        tsx (pd.DataFrame): Transaction DataFrame containing contract addresses.
        contracts (pd.DataFrame): Contract metadata indexed by contract address.
        slotMap (dict): Dictionary storing slot mapping and KNN information.
        models (dict): Dictionary where trained gas prediction models are stored.

    Returns:
        None
    """
    tsx = tsx.groupby('to_address')
    total_viables = 0 
    total_posibles = 0
    for contract_addr, contract_tsx in tqdm(tsx, desc="Contracts"): 
        if contract_addr not in contracts.index: continue 
        if contract_addr not in models: models[contract_addr] = {}
        if contract_addr not in slotMap: slotMap[contract_addr] = {}
        contract = contracts.loc[contract_addr]
        viables, posibles = process_contract(
            contract=contract,
            contract_tsx=contract_tsx,
            contract_slotMap=slotMap[contract_addr],
            contract_models=models[contract_addr]
        )
        total_viables += viables
        total_posibles += posibles
    print(f"Processed {total_viables} viable models out of the {total_posibles} possible models.")
    logger.info(f"Processed {total_viables} viable models out of the {total_posibles} possible models.")

def process_contract(
        contract: pd.Series,
        contract_tsx: pd.DataFrame,
        contract_models: dict,
        contract_slotMap: dict
) -> tuple[int, int]:
    """
    Processes all viable function signatures for a single contract.

    The function extracts transaction signatures, groups transactions by
    signature, filters out groups with insufficient samples, optionally limits
    each group size by gas distribution and launches parallel processes to handle
    each viable signature.

    Args:
        contract (pd.Series): Contract metadata for the current contract.
        contract_tsx (pd.DataFrame): Transactions associated with the contract.
        contract_models (dict): Dictionary where trained models for this contract
            are stored.
        contract_slotMap (dict): Slot mapping dictionary for this contract.

    Returns:
        tuple[int, int]: Number of successfully processed viable signatures and
        total number of possible signatures.
    """
    contract_tsx = contract_tsx.copy()
    extract_signature_column(contract_tsx)
    contract_tsx = contract_tsx.groupby('signature')
    
    m_posibles = len(contract_tsx)
    tasks = []

    for signature, signature_tsx in contract_tsx:
        if len(signature_tsx) <= Config.MIN_LEN_TSX: continue 

        truncated_tsx = limit_by_gas(signature_tsx, Config.MAX_LEN_TSX)
        current_slot_map = contract_slotMap.get(signature, {})
        tasks.append((
            contract,
            signature,
            truncated_tsx.copy(),
            current_slot_map
        ))
    
    n_viables = len(tasks)
    
    queue = Queue()
    active_processes = []
    pbar = tqdm(total=len(tasks),desc=f"Signatures {contract.name}",leave=False)
    errors_found = 0
    
    for task in tasks:
        p = Process(target=process_signature, args=(*task, queue))

        p.start()
        active_processes.append(p)

        if len(active_processes) >= MAX_PROCESSES:
            errors_found += collect_one_process(
                active_processes,
                queue,
                contract_models,
                contract_slotMap,
                pbar
            )
    
    while active_processes: 
        errors_found += collect_one_process(
            active_processes,
            queue,
            contract_models,
            contract_slotMap,
            pbar
        )
    pbar.close()
    return (n_viables-errors_found), m_posibles 

def process_signature(
        contract: pd.Series,
        signature: str,
        signature_tsx: pd.DataFrame,
        current_slotMap: dict,
        queue: Queue
) -> None:    
    """
    Processes and trains models for a specific contract-function signature.

    The function decodes transaction inputs, processes storage information and,
    depending on the configured mode, either trains gas prediction models or a KNN
    storage imputation model. Results are sent back to the parent process through
    a multiprocessing queue.

    Args:
        contract (pd.Series): Contract metadata associated with the transactions.
        signature (str): Function signature being processed.
        signature_tsx (pd.DataFrame): Transactions matching the contract and
            signature.
        current_slotMap (dict): Slot mapping information for the current
            signature.
        queue (Queue): Multiprocessing queue used to return results or errors.

    Returns:
        None
    """
    try:
        signature_tsx = process_input(signature_tsx,contract, signature)
        if not signature_tsx.empty: #couldn't decode any input
            signature_tsx,storage_columns = process_storage(signature_tsx,contract, signature, current_slotMap) 
            if Config.DO_TRAIN_NOT_KNN and len(signature_tsx) > Config.MIN_LEN_TSX:
                #signature_tsx.to_csv('data/example_preprocessed.csv', index=False, encoding='utf-8', sep=';')
                model_result = train(signature_tsx,strg_columns=storage_columns)
                queue.put({
                "ok": True,
                "type": "gas_model",
                "signature": signature,
                "model": model_result,
                "slotMap": current_slotMap["slotMap"],
                "error": None
                })      
            elif not Config.DO_TRAIN_NOT_KNN and len(signature_tsx) >=Config.MIN_LEN_KNN:
                knn_result = train_knn_storage(signature_tsx,storage_cols=storage_columns)
                if knn_result is not None:
                    queue.put({
                        "ok": True,
                        "type": "knn_storage",
                        "signature": signature,
                        "model": (knn_result, storage_columns),
                        "slotMap": current_slotMap["slotMap"],
                        "error": None
                    })
                else:
                    queue.put({
                    "ok": False,
                    "type": "skipped",
                    "signature": signature,
                    "model": None,
                    "slotMap": None,
                    "error": f"KNN models couldn't be trained."
                }) 
            else:
                queue.put({
                    "ok": False,
                    "type": "skipped",
                    "signature": signature,
                    "model": None,
                    "slotMap": None,
                    "error": f"Insuficient samples for training after removing invalid storage values."
                })
        else:
            queue.put({
                "ok": False,
                "type": "skipped",
                "signature": signature,
                "model": None,
                "slotMap": None,
                "error": f"Insuficient samples for training after removing invalid input values."
            })
    except Exception as e:
        queue.put({
            "ok": False,
            "type": "error",
            "signature": signature,
            "model": None,
            "slotMap": None,
            "error": repr(e)
        })

def collect_one_process(
        active_processes: list[Process],
        queue: Queue,
        contract_models: dict,
        contract_slotMap: dict,
        pbar: tqdm
) -> int:
    """
    Waits for one active worker process and collects its available results.

    The function removes the oldest active process from the active process list,
    periodically reads completed results from the queue while the process is
    still running, waits for the process to finish and collects any remaining
    queue messages.

    Args:
        active_processes (list[Process]): List of currently active worker
            processes.
        queue (Queue): Multiprocessing queue containing worker results.
        contract_models (dict): Dictionary where trained gas models are stored.
        contract_slotMap (dict): Dictionary where slot maps and KNN metadata are
            stored.
        pbar (tqdm): Progress bar updated as signature results are collected.

    Returns:
        int: Number of errors or skipped signatures collected from the process.
    """
    errors_found = 0
    p = active_processes.pop(0)
    while p.is_alive():
        errors_found += collect_queue_results(queue, contract_models, contract_slotMap, pbar)
        p.join(timeout=0.1)
    p.join()
    errors_found += collect_queue_results(queue, contract_models, contract_slotMap, pbar)
    return errors_found

def collect_queue_results(
        queue: Queue,
        contract_models: dict,
        contract_slot_map: dict,
        pbar: tqdm
) -> int:
    """
    Collects and integrates all available results from the multiprocessing queue.

    The function reads worker outputs from the queue, stores trained gas models,
    updates slot maps and KNN metadata, logs skipped or failed signatures and
    updates the progress bar for each processed result.

    Args:
        queue (Queue): Multiprocessing queue containing results from worker
            processes.
        contract_models (dict): Dictionary where trained gas prediction models
            are stored.
        contract_slot_map (dict): Dictionary where slot maps, storage columns,
            KNN models and metrics are stored.
        pbar (tqdm): Progress bar updated after each processed queue result.

    Returns:
        int: Number of errors or skipped signatures found in the collected
        results.
    """
    errors_found = 0
    while True:
        try: result = queue.get_nowait()
        except Empty:break

        signature = result["signature"]
        if result["ok"]:
            result_type = result.get("type")
            if result_type == "knn_storage":
                knn_result, storage_columns = result["model"]
                if signature not in contract_slot_map:
                    contract_slot_map[signature] = {
                        "columns": [],
                        "slotMap": {},
                        "knn": None,
                        "metrics": None
                    }
                contract_slot_map[signature]["slotMap"] = result["slotMap"]
                contract_slot_map[signature]["columns"] = storage_columns
                if knn_result is not None:
                    contract_slot_map[signature]["knn"] = knn_result.get("model")
                    contract_slot_map[signature]["metrics"] = knn_result.get("metrics")
                else:
                    contract_slot_map[signature]["knn"] = None
                    contract_slot_map[signature]["metrics"] = None
            elif result_type == "gas_model":
                if signature not in contract_models:
                    contract_models[signature] = result["model"]
                else:
                    antiguos = contract_models[signature]
                    nuevos = result["model"]
                    contract_models[signature] = nuevos
                    
                    contract_models[signature]['modelos_usados']= list(set(antiguos['modelos_usados'] + nuevos['modelos_usados']))
                    contract_models[signature]['storage info']= list(set(antiguos['storage info'] + nuevos['storage info']))
                    contract_models[signature]['with storage'] = {**nuevos['with storage'], **antiguos['with storage']}
                    contract_models[signature]['no storage'] = {**nuevos['no storage'], **antiguos['no storage']}
                if  signature not in contract_slot_map:
                    contract_slot_map[signature] = {
                        "columns": [],
                        "slotMap": {},
                        "knn": None,
                        "metrics": None
                    }
                contract_slot_map[signature]["slotMap"] = result["slotMap"]     
            else:
                logger.warning(f"[ERROR] Unknown signature type {signature}: {result_type}")
                errors_found += 1
        else:
            if result.get("type") == "skipped":
                logger.info(f"[SKIPPED] Signature {result['signature']}: {result['error']}")
            else:
                logger.warning(f"[ERROR] Signature {result['signature']}: {result['error']}")
            errors_found += 1

        pbar.update(1)
    return errors_found

def limit_by_gas(df, max_size=200000, bins=10):
    """
    Reduces a transaction dataset using stratified sampling by gas usage.

    The function preserves the distribution of `receipt_gas_used` by dividing
    the dataset into quantile-based gas bins and sampling proportionally from
    each bin until the target maximum size is reached.

    Args:
        df (pd.DataFrame): Transaction DataFrame containing the
            `receipt_gas_used` column.
        max_size (int, optional): Maximum number of rows to keep. Defaults to
            200000.
        bins (int, optional): Number of quantile bins used for stratification.
            Defaults to 10.

    Returns:
        pd.DataFrame: Stratified sample of the input DataFrame, or the original
        DataFrame if its size is already below the maximum.
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
            min(len(x), max(1, int(len(x) * max_size / len(df)))),
            random_state=42
        )
    )

    return sampled.drop(columns=["gas_bin"])

