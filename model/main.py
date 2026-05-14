import os
import pandas as pd
import logging
from tqdm import tqdm
from model.data_processing.reader import read, read_all, read_merged, read_reduced
from model.preprocessing.process import process_abi, extract_signature_column, process_input
from model.utils.parallel_execution import process
from multiprocessing import Manager
from model.export.dumper import export_models, import_models, export_slotMap, convert_to_dict
from model.config.config import Config, record_run
# Quitar..
from typing import Tuple
import json

logging.basicConfig(
    filename='model.log',
    level=logging.INFO, # nivel de profundidad del log DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL
    filemode='a' # 'w' para truncar y 'a' para añadir de 0
)

logger = logging.getLogger(__name__)

def main() -> None:
    logger.info(f"{'-'*15}{Config.TYPE}{'-'*15}\n")
    if(Config.ONLY_MERGE):
        print("[1/2] Only merging input information with storage data and creating the corresponding CSV files")
        read_all()
        return
    if(Config.ONLY_REDUCE):
        print("[1/2] Only reducing the dataset by removing signatures with few transactions and creating the corresponding CSV files")
        read_merged()
        return
    
    print("[1/3] Reading data.")
    tsx, contracts, slotMap  = read_all()
    #tsx, contracts, slotMap  = read()
    #tsx, contracts, slotMap = read_merged()
    #tsx, contracts, slotMap = read_reduced()
    
    contracts = process_abi(contracts)
    
    if (Config.TRAIN_NEW): models ={}
    else: models = import_models('data/models.pkl')

    if Config.DO_TRAIN_NOT_KNN: print("[2/3] Data preprocessing and model training.")
    else: print("[2/3] Data preprocessing and KNN training.")
    process(tsx, contracts, slotMap, models)

    print("[3/3] Exporting models and updating slotMap.")
    final_slotMap = convert_to_dict(slotMap)
    export_slotMap(final_slotMap, Config.SLOT_PATH, Config.KNN_PATH)
    export_models(models, Config.MODEL_PATH)
    return

if __name__=='__main__':
    main()
    record_run(Config)