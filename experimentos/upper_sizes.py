import time
import os
from tqdm import tqdm
import joblib
import json
import pandas as pd
import numpy as np
from model.config.config import Config
from model.data_processing.reader import read_merged
from model.modeling.trainer import train
from model.preprocessing.knn import train_knn_storage
from model.preprocessing.process import extract_signature_column, process_abi, process_input, process_storage
from model.utils.parallel_execution import limit_by_gas



def main():
    output_dir = "data/resultados_analisis"
    os.makedirs(output_dir, exist_ok=True)

    tsx, contracts, slotMap = read_merged()
    contracts = process_abi(contracts)
    extract_signature_column(tsx)


    grouped = tsx.groupby(['to_address', 'signature']).size().reset_index(name='n_transacciones')
    max_row = grouped.loc[grouped['n_transacciones'].idxmax()]
    print(f"The contract {max_row['to_address']} and signature {max_row['signature']} have the maximun number of transactions ({max_row['n_transacciones']}).")
    contract_addr = max_row['to_address']
    signature = max_row['signature']

    signature_tsx = tsx[(tsx['to_address'] == contract_addr) & (tsx['signature'] == signature)].copy()
    current_slotMap = {}
    try: contract = contracts.loc[contract_addr]
    except KeyError:
        print(f"Error: Couldn't find contract {contract_addr} in the contract dataset.")
        return
    
    total_size = int(max_row['n_transacciones'])
    metrics ={'contrato': contract_addr,
              'signatura': signature,
              'nº transacciones':total_size 
              }

    limit = 500000
    
    if total_size > limit:
        signature_tsx = limit_by_gas(signature_tsx, limit)
    t1 = time.perf_counter()
    signature_tsx = process_input(signature_tsx,contract, signature)
    t2 = time.perf_counter()
    signature_tsx,storage_columns = process_storage(signature_tsx,contract, signature, current_slotMap) 
    t3 = time.perf_counter()
    metrics['preprocess'] = [t2-t1,t3-t2]

    num_steps = 20
    
    options = np.logspace(np.log10(100), np.log10(min(limit,total_size)), num=num_steps).astype(int).tolist()
    upper_limit = int(np.sqrt(Config.MAX_LEN_KNN))
    raw_options = np.logspace(np.log10(1), np.log10(upper_limit), num=num_steps).astype(int).tolist()
    optionsk = sorted(list(set([x if x % 2 != 0 else x + 1 for x in raw_options])))
    
    metrics['divisiones'] = options
    metrics['divisionesk'] = optionsk
    metrics['metricas'] = []
    metrics['metricask'] = []
    metrics['tiempos'] = []
    
    for i in tqdm(range(len(options)), desc="KNN training"):
        inicio = time.perf_counter() 
        knn_result = train_knn_storage(signature_tsx,storage_cols=storage_columns, limitKNN = options[i] )
        fin = time.perf_counter() 
        metrics['metricas'].append( knn_result.get("metrics", None))
        metrics['tiempos'].append(fin - inicio)
        knn_resultK = train_knn_storage(signature_tsx,storage_cols=storage_columns, n_neighbors = options[i])
        metrics['metricask'].append( knn_resultK.get("metrics", None))
    
    with open(f'{output_dir}/metricsKNN.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("KNN Metrics Saved")
    
    
    options2 = np.logspace(np.log10(5000), np.log10(min(limit,total_size)), num=num_steps).astype(int)
    options2 = options2[:-2]
    optionsS = np.logspace(np.log10(100), np.log10(25000), num=num_steps).astype(int)
    models = {}
    models['tiempos'] = []
    for i in tqdm(range(len(options2)), desc="Big Model training"): 
        truncated_tsx = limit_by_gas(signature_tsx, options2[i])
        inicio = time.perf_counter()
        models[options2[i]] = train(truncated_tsx,strg_columns=storage_columns)
        fin = time.perf_counter() 
        models['tiempos'].append(fin - inicio)
    
    modelsS = {}
    modelsS['tiempos'] = []
    for i in tqdm(range(len(optionsS)), desc="Small Model training"): 
        truncated_tsx = limit_by_gas(signature_tsx, optionsS[i])
        inicio = time.perf_counter()
        modelsS[optionsS[i]] = train(truncated_tsx,strg_columns=storage_columns)
        fin = time.perf_counter() 
        modelsS['tiempos'].append(fin - inicio)
    
    def clean_dict_keys(obj):   
        if isinstance(obj, dict):
            return {k: clean_dict_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_dict_keys(i) for i in obj]
        elif str(type(obj)) == "<class 'dict_keys'>":
            return list(obj)
        return obj
    models = clean_dict_keys(models)
    ruta_modelos = f'{output_dir}/todos_los_modelos.joblib'
    joblib.dump(models, ruta_modelos, compress=3)
    modelsS = clean_dict_keys(modelsS)
    ruta_modelosS = f'{output_dir}/todos_los_modelosSmall.joblib'
    joblib.dump(modelsS, ruta_modelosS, compress=3)
    print("Model metrics saved")
    
    print(f"Process finished. Files saved in: {output_dir}")

    

if __name__ == "__main__":
    main()