import json
import os
import pickle
from multiprocessing.managers import DictProxy
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.config.config import Config

N_FILAS_ARCHIVO = 100000

def save_dataFrame(
        tsx: pd.DataFrame, 
        file_path: str,):
    """
    Saves a DataFrame into multiple CSV files.

    The function splits the input DataFrame into chunks of a fixed maximum
    number of rows and saves each chunk as a separate CSV file in the specified
    output directory.

    Args:
        tsx (pd.DataFrame): DataFrame to be exported.
        file_path (str): Directory where the CSV files will be saved.
    """
    print("Saving dataframe")
    if not os.path.exists(file_path): os.makedirs(file_path)

    num_partes = int(np.ceil(len(tsx) / N_FILAS_ARCHIVO))
    
    for i in tqdm(range(num_partes), desc="Saving .csv files"):
        inicio = i * N_FILAS_ARCHIVO
        fin = (i + 1) * N_FILAS_ARCHIVO
        nombre_csv = os.path.join(file_path, f"merged_part_{i}.csv")
        tsx.iloc[inicio:fin].to_csv(nombre_csv, index=False, sep=';')
    print(f"Backup CSV files saved to: {file_path}")

def export_models(
        models: DictProxy, 
        file_path: str,):
    """
    Exports trained gas prediction models to disk using pickle.

    The function converts the shared multiprocessing dictionary into a standard
    Python dictionary and serializes it to the specified file path, only when
    gas prediction models are configured to be trained.

    Args:
        models (DictProxy): Shared dictionary containing the trained models.
        file_path (str): Path where the serialized models will be saved.

    """
    if Config.DO_TRAIN_NOT_KNN:
        models = {key : value.copy() for key,value in models.items()}
        with open(file_path, 'wb') as f:
            pickle.dump(models, f)
            print(f"Saved {len(models)} gas prediction models")

def import_models(
        file_path: str
) -> Dict:
    """
    Loads serialized gas prediction models from disk.

    The function reads a pickle file containing previously trained models and
    returns them as a standard Python dictionary.

    Args:
        file_path (str): Path of the pickle file containing the exported models.

    Returns:
        Dict: Dictionary containing the loaded gas prediction models.
    """
    with open(file_path, 'rb') as f:
        models_dict = pickle.load(f)

    return dict(models_dict)

def export_slotMap(slotMap: dict, file_map: str, file_knn: str = ""):
    """
    Exports slot mapping information and KNN storage models to disk.

    The function separates serializable slot mapping information from trained KNN
    models. Slot columns, slot mappings and metrics are saved to a JSON file,
    while KNN models are optionally saved to a pickle file depending on the
    configured training mode.

    Args:
        slotMap (dict): Nested dictionary containing slot mappings, metrics and
            optional KNN models organized by contract and function signature.
        file_map (str): Path where the slot map JSON file will be saved.
        file_knn (str, optional): Path where the serialized KNN models will be
            saved. Defaults to an empty string.
    """
    slotmap_json = {}
    slotmap_knn = {}
    for contract, signatures in slotMap.items():
        slotmap_json[contract] = {}
        slotmap_knn[contract] = {}
        for signature, data in signatures.items():
            columns = data.get("columns", [])
            mapping = data.get("slotMap", {})
            knn = data.get("knn", None)
            metrics = data.get("metrics", None)
            slotmap_json[contract][signature] = {
                "columns": columns,
                "slotMap": mapping,
                "metrics": metrics
            }
            if knn is not None:
                if contract not in slotmap_knn:
                    slotmap_knn[contract] = {}
                slotmap_knn[contract][signature] = knn
                
    with open(file_map, "w", encoding="utf-8") as f:
        json.dump(slotmap_json, f, indent=4)
        print(f"Saved {len(slotmap_json)} contract slot maps to JSON")
    
    for contract in slotmap_json.keys():
        for signature, datos_modelo in signatures.items():
            knn_metrics = datos_modelo['metrics']

    if not Config.DO_TRAIN_NOT_KNN:#Queremos guardarlos
        with open(file_knn, "wb") as f:
            pickle.dump(slotmap_knn, f)
            print(f"Saved {len(slotmap_knn)} KNN models")

def convert_to_dict(proxy_obj):
    """
    Recursively converts multiprocessing proxy objects into standard Python objects.

    The function transforms DictProxy objects and nested proxy structures into
    regular dictionaries and lists, making them serializable and easier to handle
    outside multiprocessing contexts.

    Args:
        proxy_obj: Object to convert. It may be a DictProxy, dictionary, list or
            any other Python object.

    Returns:
        Any: Converted object with DictProxy instances replaced by standard
        Python dictionaries.
    """
    if isinstance(proxy_obj, (dict, DictProxy)):
        return {k: convert_to_dict(v) for k, v in proxy_obj.items()}
    elif isinstance(proxy_obj, list):
        return [convert_to_dict(i) for i in proxy_obj]
    else:
        return proxy_obj
    

