"""
benchmark_fase3.py
==================
Experimento 1 del TFM: Baseline Expert Mode (Phase 3)

Mide empíricamente el rendimiento máximo del sistema con modelos XGBoost nativos.

METODOLOGÍA:
  - F1-Score Macro sobre el conjunto de test hold-out (20% estratificado, random_state=42)
  - Benchmark de latencia: 1.000 paquetes procesados en batch, latencia media por paquete
  - Resultados guardados en resultados/benchmark_fase3.json

Ejecutar desde la carpeta raíz del proyecto:
  python src/benchmark_fase3.py
"""

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ajustar el path para importar módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from agente_centinela import AgenteCentinela, preparar_datos_desde_csv
from config import DATASETS_CONFIG, DATASET_DEFAULT, RESULTADOS_DIR

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [BENCHMARK] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("benchmark_fase3")

# ─── Parámetros ───────────────────────────────────────────────────────────────
N_LATENCY_SAMPLES = 1000   # Paquetes para el benchmark de latencia
RANDOM_STATE      = 42
TEST_SIZE         = 0.20   # Mismo split que usa el entrenamiento


def detectar_columnas(df: pd.DataFrame) -> tuple[str, str]:
    """
    Detecta automáticamente las columnas de etiqueta (binaria) y tipo (multiclase).
    Sigue la misma lógica que preparar_datos_desde_csv.
    """
    cols = [c.lower() for c in df.columns]
    col_original = list(df.columns)

    # Mapeo: candidatos de col_label (binaria)
    label_candidates = ["attack_label", "label", "attack"]
    # Mapeo: candidatos de col_type (multiclase)
    type_candidates  = ["attack_type", "type", "category", "attack_cat"]

    col_label, col_type = None, None
    for c in label_candidates:
        if c in cols:
            col_label = col_original[cols.index(c)]
            break
    for c in type_candidates:
        if c in cols:
            col_type = col_original[cols.index(c)]
            break

    if not col_label or not col_type:
        raise ValueError(
            f"No se encontraron columnas de etiqueta. "
            f"Columnas disponibles: {col_original[:10]}..."
        )
    return col_label, col_type


def cargar_datos_edge_iiot(ruta_csv: str, col_label: str, col_type: str):
    """
    Loader específico para Edge-IIoTset.
    El CSV tiene 56 columnas dummy (http.request.method_GET, etc.) guardadas
    como texto aunque son 0/1 numericos. pandas las lee como object y
    preparar_datos_desde_csv las descarta con select_dtypes.
    Este loader las convierte a float antes de filtrar, recuperando las 95 features
    con las que fue entrenado el modelo original.
    """
    logger.info(f"  Cargando Edge-IIoTset con coercion de columnas dummy...")
    df = pd.read_csv(ruta_csv, low_memory=False)

    from config import LEAKAGE_INDICATORS

    # Convertir todas las columnas a numerico donde sea posible
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ahora todas las dummy son float (0.0/1.0) y las categoricas son NaN
    X = df.select_dtypes(include='number')

    # Eliminar columnas de leakage y etiquetas
    cols_borrar = [col_label, col_type]
    for c in list(X.columns):
        if any(p in c.lower() for p in LEAKAGE_INDICATORS) and c not in [col_label, col_type]:
            cols_borrar.append(c)
    X = X.drop(cols_borrar, axis=1, errors='ignore')

    y = df[col_type]
    logger.info(f"  Datos cargados: {len(df):,} filas, {X.shape[1]} features")
    return X, y

def benchmark_latencia(centinela: AgenteCentinela, X_test: np.ndarray, n: int = N_LATENCY_SAMPLES) -> dict:
    """
    Mide la latencia de inferencia procesando `n` paquetes individuales.
    Devuelve media, P50, P95 y P99 en microsegundos (µs).
    """
    idx = np.random.choice(len(X_test), size=n, replace=(len(X_test) < n))
    muestra = X_test[idx]

    tiempos = []
    for i in range(n):
        paquete = muestra[i : i + 1]
        t0 = time.perf_counter()
        centinela.predecir(paquete)
        t1 = time.perf_counter()
        tiempos.append((t1 - t0) * 1_000_000)

    arr = np.array(tiempos)
    return {
        "n_muestras":   n,
        "media_us":     round(float(arr.mean()), 3),
        "p50_us":       round(float(np.percentile(arr, 50)), 3),
        "p95_us":       round(float(np.percentile(arr, 95)), 3),
        "p99_us":       round(float(np.percentile(arr, 99)), 3),
        "min_us":       round(float(arr.min()), 3),
        "max_us":       round(float(arr.max()), 3),
    }


def evaluar_dataset(dataset_id: str, config: dict) -> dict | None:
    """
    Evalúa un único dataset: carga el modelo, reconstruye el test set y mide métricas.
    """
    ruta_csv    = config.get("csv")
    ruta_modelo = config.get("modelo")
    col_label   = config.get("col_label")
    col_type    = config.get("col_type")

    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_id}")

    # ── Verificar existencia de archivos ──────────────────────────────────────
    if not ruta_modelo or not Path(ruta_modelo).exists():
        logger.warning(f"  Modelo no encontrado: {ruta_modelo} → SALTANDO")
        return None
    if not ruta_csv or not Path(ruta_csv).exists():
        logger.warning(f"  CSV no encontrado: {ruta_csv} → SALTANDO")
        return None

    # ── Detectar columnas si no están en config ───────────────────────────────
    logger.info(f"  Cargando CSV: {Path(ruta_csv).name}")
    if not col_label or not col_type:
        df_raw = pd.read_csv(ruta_csv, low_memory=False, nrows=5)
        col_label, col_type = detectar_columnas(df_raw)
        logger.info(f"  Columnas autodetectadas → label='{col_label}', type='{col_type}'")
    else:
        logger.info(f"  Columnas de config → label='{col_label}', type='{col_type}'")

    # ── Preparar X, y con el preprocesado unificado ───────────────────────────
    # preparar_datos_desde_csv ya maneja bool→float para Edge-IIoTset
    X, y = preparar_datos_desde_csv(
        str(ruta_csv),
        col_label=col_label,
        col_type=col_type
    )

    # ── Mismo split exacto que el entrenamiento (random_state=42, stratify) ──
    y_encoded_temp = pd.Categorical(y).codes  # encoding temporal solo para el split
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded_temp
    )
    logger.info(f"  Test set: {len(X_test):,} muestras")

    # ── Cargar el modelo entrenado ────────────────────────────────────────────
    logger.info(f"  Cargando modelo: {Path(ruta_modelo).name}")
    centinela = AgenteCentinela(modelo_path=str(ruta_modelo))
    features_esperadas = centinela.features_esperadas

    # ── Alinear columnas al orden exacto que el modelo espera ─────────────────
    # Igual que hace el Orquestador en producción (CASO A):
    # las features que faltan se rellenan con 0.0 (valor neutral para XGBoost).
    features_lista = []
    for f in features_esperadas:
        if f in X_test.columns:
            features_lista.append(X_test[f].values)
        else:
            features_lista.append(np.zeros(len(X_test)))
            logger.debug(f"  Feature '{f}' no encontrada en test set → relleno con 0.0")

    X_test_alineado = np.column_stack(features_lista)
    n_missing = sum(1 for f in features_esperadas if f not in X_test.columns)
    if n_missing > 0:
        logger.warning(f"  {n_missing} features rellenadas con 0.0 (ausentes en test set)")

    # ── Métricas de clasificación ─────────────────────────────────────────────
    logger.info("  Calculando métricas sobre test set...")
    y_pred_num = centinela.modelo.predict(X_test_alineado)
    y_test_num = centinela.label_encoder.transform(y_test)

    f1_macro   = f1_score(y_test_num, y_pred_num, average="macro",    zero_division=0)
    f1_weighted= f1_score(y_test_num, y_pred_num, average="weighted", zero_division=0)
    precision  = precision_score(y_test_num, y_pred_num, average="macro", zero_division=0)
    recall     = recall_score(y_test_num, y_pred_num, average="macro",    zero_division=0)
    accuracy   = accuracy_score(y_test_num, y_pred_num)
    n_clases   = len(centinela.clases)

    logger.info(f"  Accuracy:     {accuracy:.4f}")
    logger.info(f"  F1-Macro:     {f1_macro:.4f}")
    logger.info(f"  Precision:    {precision:.4f}")
    logger.info(f"  Recall:       {recall:.4f}")
    logger.info(f"  Clases:       {n_clases}")

    # ── Benchmark de latencia ─────────────────────────────────────────────────
    logger.info(f"  Midiendo latencia ({N_LATENCY_SAMPLES} inferencias individuales)...")
    latencia = benchmark_latencia(centinela, X_test_alineado)
    logger.info(f"  Latencia media: {latencia['media_us']:.1f} µs/paquete")
    logger.info(f"  Latencia P95:   {latencia['p95_us']:.1f} µs/paquete")

    return {
        "dataset":      dataset_id,
        "n_test":       len(X_test),
        "n_clases":     n_clases,
        "accuracy":     round(accuracy, 4),
        "f1_macro":     round(f1_macro, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "latencia":     latencia,
    }


def main():
    logger.info("BENCHMARK FASE 3 — Expert Mode (XGBoost nativo)")
    logger.info(f"N inferencias para latencia: {N_LATENCY_SAMPLES}")

    resultados = []

    for dataset_id, config in DATASETS_CONFIG.items():
        if not config:  # Saltamos Autodetección (sin CSV ni modelo)
            continue
        resultado = evaluar_dataset(dataset_id, config)
        if resultado:
            resultados.append(resultado)

    if not resultados:
        logger.error("No se pudo evaluar ningún dataset. Verifica los modelos y CSVs.")
        return

    # ── Resumen final ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN — Copiar a la Tabla II de la memoria:")
    logger.info(f"{'Dataset':<15} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10} {'Latencia (avg)':>16}")
    logger.info("-" * 65)
    for r in resultados:
        lat = r["latencia"]["media_us"]
        logger.info(
            f"{r['dataset']:<15} {r['f1_macro']:>10.4f} "
            f"{r['precision_macro']:>10.4f} {r['recall_macro']:>10.4f} "
            f"{lat:>13.1f} µs"
        )

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    ruta_salida = RESULTADOS_DIR / "benchmark_fase3.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados guardados en: {ruta_salida}")


if __name__ == "__main__":
    main()
