"""
generar_features_comunes.py
============================
Calcula el espacio de features comunes entre Edge-IIoTset, TON_IoT y Bot-IoT
mediante un mapa de equivalencias semánticas, y entrena el modelo Fallback
universal sobre ese subconjunto de features extraídas directamente de los CSVs.

Uso:
    cd src
    python generar_features_comunes.py

Salida:
    - data/features_comunes.json   → lista de features canónicas comunes
    - modelos/xgboost_fallback_model.pkl → modelo fallback reentrenado
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Rutas relativas al repo
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import DATASETS_DIR, MODELOS_DIR, DATA_DIR
from agente_centinela import AgenteCentinela

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MAPA DE EQUIVALENCIAS SEMÁNTICAS
# Clave: nombre canónico (feature universal)
# Valor: dict con el nombre de columna en cada dataset (None si no existe)
# ---------------------------------------------------------------------------
MAPA_EQUIVALENCIAS = {
    # --- Volumen de tráfico ---
    "bytes_src": {
        "Edge-IIoT": "tcp.len",          # bytes payload TCP/UDP
        "TON_IoT":   "src_bytes",
        "Bot-IoT":   "srate",            # send rate como proxy de volumen
    },
    "bytes_dst": {
        "Edge-IIoT": None,               # no directo en Edge-IIoT
        "TON_IoT":   "dst_bytes",
        "Bot-IoT":   "drate",
    },
    "pkts_src": {
        "Edge-IIoT": "udp.stream",       # ID de stream / paquetes UDP
        "TON_IoT":   "src_pkts",
        "Bot-IoT":   "N_IN_Conn_P_SrcIP",
    },
    "pkts_dst": {
        "Edge-IIoT": "dns.retransmission", # Como proxy de actividad de red
        "TON_IoT":   "dst_pkts",
        "Bot-IoT":   "N_IN_Conn_P_DstIP",
    },
    "avg_packet_size": {
        "Edge-IIoT": "tcp.len",          # En el entrenamiento se usará como base
        "TON_IoT":   None,               # Se calculará dinámicamente
        "Bot-IoT":   None,
    },
    # --- Puertos ---
    "puerto_src": {
        "Edge-IIoT": None,               # no en Edge-IIoT (Tshark captura nivel 4+)
        "TON_IoT":   "src_port",
        "Bot-IoT":   "sport",
    },
    "puerto_dst": {
        "Edge-IIoT": None,
        "TON_IoT":   "dst_port",
        "Bot-IoT":   "dport",
    },
    # --- Tiempo / duración ---
    "duracion": {
        "Edge-IIoT": "udp.time_delta",
        "TON_IoT":   "duration",
        "Bot-IoT":   "mean",             # duración media de paquetes
    },
    # --- Flags TCP ---
    "flag_syn": {
        "Edge-IIoT": "tcp.connection.syn",
        "TON_IoT":   None,
        "Bot-IoT":   None,
    },
    "flag_fin": {
        "Edge-IIoT": "tcp.connection.fin",
        "TON_IoT":   None,
        "Bot-IoT":   None,
    },
    "flag_rst": {
        "Edge-IIoT": "tcp.connection.rst",
        "TON_IoT":   None,
        "Bot-IoT":   None,
    },
    # --- DNS ---
    "dns_tipo": {
        "Edge-IIoT": "dns.qry.type",
        "TON_IoT":   "dns_qtype",
        "Bot-IoT":   None,
    },
    # --- HTTP ---
    "http_codigo": {
        "Edge-IIoT": "http.response",
        "TON_IoT":   "http_status_code",
        "Bot-IoT":   None,
    },
    "http_bytes_resp": {
        "Edge-IIoT": "http.content_length",
        "TON_IoT":   "http_response_body_len",
        "Bot-IoT":   None,
    },
    # --- ICMP ---
    "icmp_seq": {
        "Edge-IIoT": "icmp.seq_le",
        "TON_IoT":   None,
        "Bot-IoT":   None,
    },
    # --- Estadísticos Bot-IoT ---
    "stat_max": {
        "Edge-IIoT": None,
        "TON_IoT":   None,
        "Bot-IoT":   "max",
    },
    "stat_min": {
        "Edge-IIoT": None,
        "TON_IoT":   None,
        "Bot-IoT":   "min",
    },
    "stat_stddev": {
        "Edge-IIoT": None,
        "TON_IoT":   None,
        "Bot-IoT":   "stddev",
    },
    # --- Estado de conexión ---
    "estado_conn": {
        "Edge-IIoT": "tcp.flags",
        "TON_IoT":   "http_status_code",
        "Bot-IoT":   "state_number",
    },
}

# Columnas de etiqueta por dataset
# Índice 0: columna binaria (Normal/Ataque)
# Índice 1: columna de tipo de ataque específico (para entrenamiento multiclase)
LABEL_COLS = {
    "Edge-IIoT": ["Attack_label", "Attack_type"],
    "TON_IoT":   ["label",        "type"],
    "Bot-IoT":   ["attack",       "category"],   # 'category' tiene los tipos (DDoS, DoS, Normal...)
}

# ---------------------------------------------------------------------------
# FUNCIÓN: extraer features de un CSV usando el mapa de equivalencias
# ---------------------------------------------------------------------------

def extraer_features_dataset(csv_path: Path, dataset_id: str, features_canonicas: list) -> pd.DataFrame:
    """
    Lee el CSV y construye un DataFrame con las features canónicas,
    usando el mapa de equivalencias para renombrar columnas.
    Las features sin equivalencia en este dataset se rellenan con NaN.
    """
    logger.info(f"Cargando {dataset_id} desde {csv_path.name}...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Mejoras específicas para evitar NaNs en Edge-IIoT (Tshark es muy granular)
    if dataset_id == "Edge-IIoT":
        # Combinar bytes de diferentes protocolos en una sola columna de ayuda
        df['bytes_proxy'] = df[['tcp.len', 'udp.time_delta', 'http.content_length', 'mqtt.len', 'mbtcp.len']].max(axis=1)

    # Eliminar TODAS las columnas de etiqueta para evitar leakage
    leakage_cols = LABEL_COLS.get(dataset_id, [])
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    resultado = pd.DataFrame(index=df.index)

    for feature_canonica in features_canonicas:
        if feature_canonica == "avg_packet_size":
            continue # Se calcula al final
            
        col_original = MAPA_EQUIVALENCIAS[feature_canonica].get(dataset_id)
        if col_original and col_original in df.columns:
            serie = pd.to_numeric(df[col_original], errors="coerce")
            resultado[feature_canonica] = serie
        elif dataset_id == "Edge-IIoT" and feature_canonica == "bytes_src":
            resultado[feature_canonica] = df['bytes_proxy']
        else:
            resultado[feature_canonica] = np.nan

    # Cálculo dinámico de avg_packet_size para todos los datasets
    # Evita NaNs si tenemos bytes y paquetes
    if 'bytes_src' in resultado.columns and 'pkts_src' in resultado.columns:
        resultado['avg_packet_size'] = (resultado['bytes_src'] / resultado['pkts_src'].replace(0, 1)).fillna(0)

    logger.info(f"  → {dataset_id}: {len(resultado)} filas, {resultado.notna().any().sum()}/{len(features_canonicas)} features presentes")
    return resultado


# ---------------------------------------------------------------------------
# TAXONOMÍA CANÓNICA
# Unifica los distintos nombres de ataque de los 3 datasets en 8 clases comunes.
# Cualquier tipo no listado se mapea a "Other_Attack".
# ---------------------------------------------------------------------------
TAXONOMIA_COMUN = {
    # DDoS / DoS
    "ddos": "Ataque",    "ddos_http": "Ataque", "ddos_icmp": "Ataque",
    "ddos_tcp": "Ataque", "ddos_udp": "Ataque",  "dos": "Ataque",
    # Scanning / Reconnaissance
    "scanning": "Ataque",    "port_scanning": "Ataque",
    "fingerprinting": "Ataque", "reconnaissance": "Ataque",
    # Injection (SQL, XSS)
    "injection": "Ataque", "sql_injection": "Ataque", "xss": "Ataque",
    # Backdoor / Malware
    "backdoor": "Ataque", "ransomware": "Ataque", "uploading": "Ataque",
    # Credential attacks
    "password": "Ataque",
    # MITM
    "mitm": "Ataque",
    # Data Theft
    "theft": "Ataque",
    # Normal
    "normal": "Normal", "0": "Normal", "benign": "Normal", "legitimate": "Normal",
}


def normalizar_clase(nombre: str) -> str:
    """Mapea un nombre de clase raw a la taxonomía canónica."""
    key = str(nombre).strip().lower()
    return TAXONOMIA_COMUN.get(key, "Ataque")


def inferir_etiquetas(csv_path: Path, dataset_id: str) -> pd.Series:
    """
    Lee la columna de TIPO DE ATAQUE de cada dataset (multiclase)
    y la normaliza a la TAXONOMÍA CANÓNICA de 8 clases.
    Así el modelo Fallback predice clases coherentes entre los 3 datasets.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    cols = LABEL_COLS[dataset_id]

    type_col = cols[1] if len(cols) > 1 else cols[0]
    if type_col not in df.columns:
        type_col = cols[0]
        if type_col not in df.columns:
            raise ValueError(f"Ninguna columna de etiqueta encontrada en {dataset_id}: {cols}")

    col = df[type_col].astype(str).str.strip().apply(normalizar_clase)

    logger.info(f"  Clases canónicas en {dataset_id}: {sorted(col.unique())}")
    return col.reset_index(drop=True)



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # 1. Determinar features canónicas que tienen equivalencia en ≥ 2 datasets
    #    (ser exigente: al menos 2 de 3 para no perder demasiada información)
    datasets_ids = ["Edge-IIoT", "TON_IoT", "Bot-IoT"]
    features_validas = []
    for feature, mapa in MAPA_EQUIVALENCIAS.items():
        cobertura = sum(1 for ds in datasets_ids if mapa.get(ds) is not None)
        if cobertura >= 2:
            features_validas.append(feature)

    # avg_packet_size is computed explicitly, so we must add it
    if "avg_packet_size" not in features_validas:
        features_validas.append("avg_packet_size")

    logger.info(f"\n{'='*60}")
    logger.info(f"Features canónicas con cobertura ≥ 2/3 datasets: {len(features_validas)}")
    logger.info(f"{features_validas}")
    logger.info(f"{'='*60}\n")

    # 2. Guardar la lista en JSON (la leerá config.py)
    features_json_path = DATA_DIR / "features_comunes.json"
    with open(features_json_path, "w", encoding="utf-8") as f:
        json.dump(features_validas, f, indent=2)
    logger.info(f"Lista guardada en {features_json_path}")

    # 3. Cargar y combinar los 3 datasets con las features canónicas
    csv_paths = {
        "Edge-IIoT": DATASETS_DIR / "EdgeIIoTset.csv",
        "TON_IoT":   DATASETS_DIR / "TON_IoT.csv",
        "Bot-IoT":   DATASETS_DIR / "Bot-IoT.csv",
    }

    frames_X = []
    frames_y = []

    for ds_id, csv_path in csv_paths.items():
        if not csv_path.exists():
            logger.warning(f"CSV no encontrado: {csv_path}. Se omite {ds_id}.")
            continue

        X_ds = extraer_features_dataset(csv_path, ds_id, features_validas)
        y_ds = inferir_etiquetas(csv_path, ds_id)

        # Alinear índices
        min_len = min(len(X_ds), len(y_ds))
        frames_X.append(X_ds.iloc[:min_len].reset_index(drop=True))
        frames_y.append(y_ds.iloc[:min_len])

    if not frames_X:
        logger.error("No se pudo cargar ningún dataset. Abortando.")
        sys.exit(1)

    X_combined = pd.concat(frames_X, ignore_index=True)
    y_combined = pd.concat(frames_y, ignore_index=True)

    logger.info(f"Dataset combinado: {len(X_combined):,} muestras | {X_combined.shape[1]} features")
    logger.info(f"Distribución de clases:\n{y_combined.value_counts().to_string()}")

    # 4. Entrenar el modelo Fallback
    logger.info("\nEntrenando modelo Fallback con features comunes...")
    centinela_fallback = AgenteCentinela()
    centinela_fallback.entrenar(X_combined, y_combined)

    # 5. Guardar el modelo
    ruta_modelo = MODELOS_DIR / "xgboost_fallback_model.pkl"
    centinela_fallback.guardar_modelo(str(ruta_modelo))
    logger.info(f"Modelo Fallback guardado en: {ruta_modelo}")

    logger.info("\n✓ Proceso completado exitosamente.")
    logger.info(f"  → Features comunes: {features_validas}")
    logger.info(f"  → Modelo: {ruta_modelo}")
    logger.info(f"  → Para activar: reinicia el sistema (config.py cargará features_comunes.json automáticamente)")


if __name__ == "__main__":
    main()
