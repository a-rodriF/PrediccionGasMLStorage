import pandas as pd
import json
import os

DATA_PATH = 'data/'
TSX_FILE = os.path.join(DATA_PATH, 'example_tsx.csv')
STRG_FOLDER = os.path.join(DATA_PATH, 'strg_info')
OUTPUT_FILE = os.path.join(DATA_PATH, 'example_strg.json')


def normalize_hash(h):
    """Normaliza hash para evitar problemas de formato"""
    h = str(h).strip().lower()
    if not h.startswith('0x'):
        h = '0x' + h
    return h


def main():
    print("[1/4] Loading list of transactions from .csv file...")

    df_example = pd.read_csv(TSX_FILE, sep=';')

    if 'hash' not in df_example.columns:
        raise KeyError(
            f"The 'hash' column doesn't exist in {TSX_FILE}. "
            f"Columns: {df_example.columns.tolist()}"
        )

    hashes_buscados = set(
        df_example['hash']
        .apply(normalize_hash)
        .unique()
    )

    print(f"Total hashes in {TSX_FILE}: {len(hashes_buscados)}")

    print("[2/4] Searching for matches in the storage files...")

    archivos = [f for f in os.listdir(STRG_FOLDER) if f.endswith('.json')]
    print(f"JSON files found: {len(archivos)}")

    transacciones_encontradas = []
    for nombre_archivo in archivos:
        ruta_completa = os.path.join(STRG_FOLDER, nombre_archivo)
        try:
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                datos_json = json.load(f)

            if isinstance(datos_json, list):
                for tx in datos_json:
                    if isinstance(tx, dict):
                        tx_hash = normalize_hash(tx.get('hash', ''))
                        if tx_hash in hashes_buscados:
                            transacciones_encontradas.append(tx)

            elif isinstance(datos_json, dict):
                if 'hash' in datos_json:
                    tx_hash = normalize_hash(datos_json.get('hash', ''))
                    if tx_hash in hashes_buscados:
                        transacciones_encontradas.append(datos_json)

                else:
                    for clave in ['result', 'transactions', 'items']:
                        if clave in datos_json and isinstance(datos_json[clave], list):
                            for tx in datos_json[clave]:
                                if isinstance(tx, dict):
                                    tx_hash = normalize_hash(tx.get('hash', ''))
                                    if tx_hash in hashes_buscados:
                                        transacciones_encontradas.append(tx)

        except Exception as e:
            print(f"[WARN] Error reading {nombre_archivo}: {e}")

    print("[3/4] Removing hash duplicates...")

    transacciones_unicas = {}
    for tx in transacciones_encontradas:
        tx_hash = normalize_hash(tx.get('hash', ''))
        if tx_hash:
            transacciones_unicas[tx_hash] = tx

    resultado_final = list(transacciones_unicas.values())

    total_encontradas = len(transacciones_encontradas)
    total_unicas = len(resultado_final)
    total_buscadas = len(hashes_buscados)
    total_missing = total_buscadas - total_unicas
    cobertura = total_unicas / total_buscadas if total_buscadas > 0 else 0

    print("\n RESULTS:")
    print(f"- Matches found (including duplicates): {total_encontradas}")
    print(f"- Unique matches: {total_unicas}")
    print(f"- Hashes searched: {total_buscadas}")
    print(f"- Missing hashes: {total_missing}")
    print(f"- Coverage: {cobertura:.2%}")

    print("[4/4] Saving JSON file...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(resultado_final, f, indent=4, ensure_ascii=False)

    print(f"\n File created successfully in: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()