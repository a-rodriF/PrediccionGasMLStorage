import json
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

def normalize_hash(h: str) -> str:
    """
    Normalizes a transaction or contract hash.

    The function converts the input value to a lowercase string, removes leading
    and trailing whitespace and ensures that the hash starts with the `0x`
    prefix.

    Args:
        h (str): Hash value to normalize.

    Returns:
        str: Normalized hash in lowercase format with the `0x` prefix.
    """
    h = str(h).strip().lower()
    if h and not h.startswith('0x'):
        h = '0x' + h
    return h

def process_one_file(args):
    """
    Processes a single storage JSON file and extracts matching transaction data.

    The function reads one JSON file, checks whether it contains a list of
    transaction dictionaries and extracts the `storage_before` information for
    transactions whose hash is included in the target hash set.

    Args:
        args (tuple): Tuple containing:
            - filename (str): Name of the JSON file to process.
            - strg_path (str): Directory where the storage JSON files are stored.
            - hashes_buscados (set): Set of normalized transaction hashes to find.

    Returns:
        list[dict]: List of dictionaries containing the matched transaction hash
        and its corresponding `storage_before` information.
    """
    filename, strg_path, hashes_buscados = args
    coincidencias = []
    filepath = os.path.join(strg_path, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):#son lsitas de json
            for tx in data:#cada transacción
                if not isinstance(tx, dict):
                    continue
                tx_hash = normalize_hash(tx.get('hash', ''))
                if tx_hash in hashes_buscados:
                    #Si tenemos información pero no hay campo storage_before-> No se modifica el storage
                    coincidencias.append({
                        'hash': tx_hash,
                        'storage_before': tx.get('storage_before', {}) if tx.get('storage_before') is not None else {}
                    })

        else:
            print(f"[WARN] File is not a transaction list: {filename}")
    except MemoryError:
        print(f"[WARN] MemoryError reading {filename}")
    except Exception as e:
        print(f"[WARN] Error reading {filename}: {e}")

    return coincidencias

def extract_storage_byHash(strg_path, hashes_buscados, n_processes=4):
    """
    Extracts relevant storage information from multiple JSON files in parallel.

    The function scans all JSON files in the storage directory, processes them
    using multiple worker processes and collects the storage information for the
    transaction hashes included in the target hash set. Duplicate hashes are
    removed, keeping the last occurrence.

    Args:
        strg_path (str): Path to the directory containing storage JSON files.
        hashes_buscados (set): Set of normalized transaction hashes to search for.
        n_processes (int, optional): Number of parallel processes to use.
            Defaults to 4.

    Returns:
        pd.DataFrame: DataFrame containing the matched transaction hashes and
        their corresponding `storage_before` values.
    """
    archivos = [f for f in os.listdir(strg_path) if f.endswith('.json')]
    args = [(filename, strg_path, hashes_buscados) for filename in archivos]
    coincidencias = []

    with Pool(processes=n_processes) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_file, args),
            total=len(args),
            desc="Reading Relevant Storage Info"
        ):
            coincidencias.extend(res)

    df = pd.DataFrame(coincidencias)
    df = df.drop_duplicates(subset='hash', keep='last')
    return df



