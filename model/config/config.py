import os
from dotenv import load_dotenv
import datetime
import logging

# Carga las variables del archivo .env al entorno
load_dotenv("model/config.env")
logger = logging.getLogger(__name__)

class Config:
   # 1. Cargamos el TYPE primero para usarlo en las rutas
    TYPE = os.getenv('TYPE', '')
    RUN_PATH = os.getenv('RUN_PATH', 'runs.txt')

    # 2. Cargamos la ruta base
    DATA_PATH = os.getenv('DATA_PATH', 'data/')

    # 3. CONSTRUCCIÓN DE RUTAS: 
    TSX_PATH = os.path.join(DATA_PATH, 'tsx')
    STRG_PATH = os.path.join(DATA_PATH, 'strg_info')
    MRGD_PATH = os.path.join(DATA_PATH, 'merged')
    RDC_PATH = os.path.join(DATA_PATH, 'reduced')

    # Archivos específicos que incluyen el prefijo TYPE
    SLOT_PATH = os.path.join(DATA_PATH, f"{TYPE}slotMap.json")
    KNN_PATH = os.path.join(DATA_PATH, f"{TYPE}knn.pkl")
    MODEL_PATH = os.path.join(DATA_PATH, f"{TYPE}models.pkl")
    
    # --- Parámetros Numéricos y Booleanos ---
    N_JSON_PROCESES = int(os.getenv('N_JSON_PROCESES', 4))
    PRE_MERGE_INFO = os.getenv('PRE_MERGE_INFO', 'False').lower() in ('true', '1', 't')
    SHOW_STRG_INFO = os.getenv('SHOW_STRG_INFO', 'False').lower() in ('true', '1', 't')
    ONLY_MERGE = os.getenv('ONLY_MERGE', 'False').lower() in ('true', '1', 't')
    ONLY_REDUCE = os.getenv('ONLY_REDUCE', 'False').lower() in ('true', '1', 't')
    if ONLY_MERGE or ONLY_REDUCE: SHOW_STRG_INFO = True

    # --- Parallel Execution ---
    MIN_LEN_TSX = int(os.getenv('MIN_LEN_TSX', 100))
    MAX_LEN_TSX = int(os.getenv('MAX_LEN_TSX', 200000))
    MIN_LEN_KNN = int(os.getenv('MIN_LEN_KNN', 10))
    MAX_LEN_KNN = int(os.getenv('MAX_LEN_KNN', 500))
    TRAIN_NEW = os.getenv('TRAIN_NEW', 'False').lower() in ('true', '1', 't')
    DO_TRAIN_NOT_KNN = os.getenv('DO_TRAIN_NOT_KNN', 'False').lower() in ('true', '1', 't')

    # --- Preprocess ---
    STRG_MANAGEMENT = int(os.getenv('STRG_MANAGEMENT', 0))
    SAVE_HASH_MISSING_INFO = os.getenv('SAVE_HASH_MISSING_INFO', 'False').lower() in ('true', '1', 't')
    if not DO_TRAIN_NOT_KNN and STRG_MANAGEMENT != 0:
        STRG_MANAGEMENT = 0

def record_run(config_class):
    """
    Escribe la configuración actual en el archivo definido en RUN_PATH.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = (
        f"{'='*60}\n"
        f"RUN DATE: {timestamp}\n"
        f"TYPE:     {getattr(config_class, 'TYPE', 'UNDEFINED')}\n"
        f"{'-'*60}\n"
        f"Paths:\n"
        f"  - DATA_PATH:   {config_class.DATA_PATH}\n"
        f"  - TSX_PATH:    {config_class.TSX_PATH}\n"
        f"  - STRG_PATH:   {config_class.STRG_PATH}\n"
        f"  - MRGD_PATH:   {config_class.MRGD_PATH}\n"
        f"  - RDC_PATH:    {config_class.RDC_PATH}\n"
        f"  - SLOT_PATH:   {config_class.SLOT_PATH}\n"
        f"  - KNN_PATH:    {config_class.KNN_PATH}\n"
        f"  - MODEL_PATH:  {config_class.MODEL_PATH}\n"
        f"Execution Params:\n"
        f"  - CORES:       {config_class.N_JSON_PROCESES}\n"
        f"  - TSX RANGE:   [{config_class.MIN_LEN_TSX} - {config_class.MAX_LEN_TSX}]\n"
        f"  - KNN RANGE:   [{config_class.MIN_LEN_KNN} - {config_class.MAX_LEN_KNN}]\n"
        f"Logic Flags:\n"
        f"  - ONLY_MERGE:     {config_class.ONLY_MERGE}\n"
        f"  - ONLY_REDUCE:    {config_class.ONLY_REDUCE}\n"
        f"  - PRE_MERGE_INFO: {config_class.PRE_MERGE_INFO}\n"
        f"  - SHOW_STRG_INFO: {config_class.SHOW_STRG_INFO}\n"
        f"  - TRAIN_NEW:      {config_class.TRAIN_NEW}\n"
        f"  - TRAIN_NOT_KNN:  {config_class.DO_TRAIN_NOT_KNN}\n"
        f"  - STRG_MGMT:      {config_class.STRG_MANAGEMENT}\n"
        f"  - SAVE_MISSING:   {config_class.SAVE_HASH_MISSING_INFO}\n"
        f"{'='*60}\n\n"
    )
    try:
        with open(config_class.RUN_PATH, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"Process completed successfully. ")
    except Exception as e:
        print(f"Error.")