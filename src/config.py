"""Configuración centralizada del sistema."""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Directorios
SRC_DIR = BASE_DIR
DATA_DIR = PROJECT_ROOT / "data"
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
DATASETS_DIR = DATA_DIR
MODELOS_DIR = PROJECT_ROOT / "modelos"

# Archivos y Multi-Dataset
MITRE_KB = DATA_DIR / "mitre_capec_kb.txt"

# Arquitectura Híbrida de Modelos
MODELO_ANALISTA = "llama-3.3-70b-versatile"    # Inteligencia superior para análisis
MODELO_NORMALIZADOR = "llama-3.3-70b-versatile" # Velocidad y alta capacidad cognitiva para esquemas complejos
MODELO_RESPONDEDOR = "llama-3.1-8b-instant"    # Equilibrio para mitigaciones
DEFAULT_LLM_MODEL = MODELO_NORMALIZADOR

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES COMUNES — Intersección semántica entre los 3 datasets
# ─────────────────────────────────────────────────────────────────────────────
# Calculadas por generar_features_comunes.py a partir del mapa de equivalencias
# entre Edge-IIoTset, TON_IoT y Bot-IoT.
# ─────────────────────────────────────────────────────────────────────────────
_FEATURES_COMUNES_PATH = DATA_DIR / "features_comunes.json"
try:
    with open(_FEATURES_COMUNES_PATH, "r", encoding="utf-8") as _f:
        FEATURES_COMUNES = json.load(_f)
except FileNotFoundError:
    FEATURES_COMUNES = None
except Exception:
    FEATURES_COMUNES = None

# Features que el Normalizador LLM debe mapear y el modelo Fallback espera recibir.
FEATURES_FALLBACK = FEATURES_COMUNES

# Dataset por Defecto (Base)
DATASET_DEFAULT = "Edge-IIoT"

# Resolución flexible de la ruta del dataset principal:
# Si el dataset completo (EdgeIIoTset.csv ~976MB) no está presente,
# se utiliza automáticamente el sample representativo incluido (sample_EdgeIIoTset.csv).
_EDGE_CSV = DATASETS_DIR / "EdgeIIoTset.csv"
if not _EDGE_CSV.exists() and (DATASETS_DIR / "sample_EdgeIIoTset.csv").exists():
    _EDGE_CSV = DATASETS_DIR / "sample_EdgeIIoTset.csv"

# Modelo de Fallback Universal (Fase 1 — Modo Supervivencia)
# Entrenado con FEATURES_COMUNES (intersección canónica entre los 3 datasets).
MODELO_FALLBACK = MODELOS_DIR / "xgboost_fallback_model.pkl"

# ID especial para el modo de Autodetección (sin contexto de red conocido)
AUTODETECCION_ID = "Autodetección"

# Diccionario de configuraciones por Dataset
DATASETS_CONFIG = {
    # Entrada especial: sin CSV ni modelo. El Orquestador enruta directamente
    # al Normalizador LLM + Modelo Fallback Universal (Modo Supervivencia).
    AUTODETECCION_ID: {},
    "Edge-IIoT": {
        "csv": _EDGE_CSV,
        "modelo": MODELOS_DIR / "xgboost_model.pkl",
        "modelo_fallback": MODELO_FALLBACK,
        "col_label": "Attack_label",
        "col_type": "Attack_type"
    },
    "TON_IoT": {
        "csv": DATASETS_DIR / "TON_IoT.csv",
        "modelo": MODELOS_DIR / "xgboost_model_TON_IoT.pkl",
    },
    "Bot-IoT": {
        "csv": DATASETS_DIR / "Bot-IoT.csv",
        "modelo": MODELOS_DIR / "xgboost_model_Bot_IoT.pkl",
    }
}

# Indicadores de Fuga de Datos (Data Leakage)
# Columnas que suelen contener la respuesta o IDs que correlacionan perfectamente con el ataque
LEAKAGE_INDICATORS = [
    'category', 'subcategory', 'target', 'class', 'label', 'id_', 
    'pkseqid', 'seq', 'ts', 'te', 'stime', 'ltime', 'saddr', 'daddr'
]

# RAG y caché
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
LEARNED_SCHEMAS_PATH = DATA_DIR / "learned_schemas.json"

# Por retrocompatibilidad mantenemos estas variables apuntando al default
DATASET_LIMPIO = DATASETS_CONFIG[DATASET_DEFAULT]["csv"]
MODELO_CENTINELA = DATASETS_CONFIG[DATASET_DEFAULT]["modelo"]
COL_LABEL = DATASETS_CONFIG[DATASET_DEFAULT]["col_label"]
COL_TYPE = DATASETS_CONFIG[DATASET_DEFAULT]["col_type"]

# Crear directorios si no existen para evitar errores de escritura
for folder in [RESULTADOS_DIR, MODELOS_DIR, FAISS_INDEX_DIR]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
