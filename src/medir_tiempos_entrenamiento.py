"""
medir_tiempos_entrenamiento.py
==============================
Mide empíricamente el tiempo de entrenamiento XGBoost para cada dataset
(Edge-IIoTset, TON_IoT, Bot-IoT) y guarda los resultados en JSON.

Ejecutar desde la raíz del proyecto:
    python src/medir_tiempos_entrenamiento.py

Los resultados se usan para rellenar la Tabla V de la memoria (Phase 2
Asynchronous Training wall-clock times).
"""

import sys
import time
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from agente_centinela import AgenteCentinela, preparar_datos_desde_csv
from config import DATASETS_CONFIG

RESULTADOS_PATH = Path(__file__).parent.parent / "resultados" / "tiempos_entrenamiento.json"


def medir_dataset(nombre: str, config: dict) -> dict:
    """Mide el tiempo de carga + entrenamiento para un dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {nombre}")
    logger.info(f"{'='*60}")

    # --- Carga ---
    t0_carga = time.perf_counter()
    logger.info(f"Cargando {config['csv'].name} ...")
    col_label = config.get("col_label", "label")
    col_type  = config.get("col_type",  None)

    try:
        X, y = preparar_datos_desde_csv(
            str(config["csv"]),
            col_label=col_label,
            col_type=col_type,
        )
    except TypeError:
        # Versiones sin col_type
        X, y = preparar_datos_desde_csv(str(config["csv"]), col_label=col_label)

    t1_carga = time.perf_counter()
    carga_s  = t1_carga - t0_carga

    n_total   = len(X)
    n_features = X.shape[1]
    logger.info(f"  Cargado: {n_total:,} filas, {n_features} features  ({carga_s:.1f}s)")
    logger.info(f"  Clases ({y.nunique()}): {sorted(y.unique().tolist())}")

    # --- Entrenamiento ---
    logger.info("Entrenando XGBoost ...")
    centinela = AgenteCentinela()

    t0_train = time.perf_counter()
    metricas = centinela.entrenar(X, y)
    t1_train = time.perf_counter()
    train_s  = t1_train - t0_train

    train_min = train_s / 60

    logger.info(f"  Entrenamiento completado en {train_s:.1f}s  ({train_min:.2f} min)")
    logger.info(f"  Acc={metricas['accuracy']:.4f}  F1-M={metricas['f1_macro']:.4f}")
    logger.info(f"  Train samples: {metricas['num_train_samples']:,}")
    logger.info(f"  Test  samples: {metricas['num_test_samples']:,}")

    return {
        "dataset":          nombre,
        "n_total":          n_total,
        "n_train":          metricas["num_train_samples"],
        "n_test":           metricas["num_test_samples"],
        "n_features":       n_features,
        "carga_s":          round(carga_s, 2),
        "train_s":          round(train_s, 2),
        "train_min":        round(train_min, 3),
        "accuracy":         round(metricas["accuracy"], 4),
        "f1_macro":         round(metricas["f1_macro"], 4),
    }


def main():
    logger.info("MEDICIÓN DE TIEMPOS DE ENTRENAMIENTO — Fase 2 (Tabla V TFM)")

    ORDEN = ["Edge-IIoT", "TON_IoT", "Bot-IoT"]
    resultados = {}

    for nombre in ORDEN:
        if nombre not in DATASETS_CONFIG:
            logger.warning(f"  '{nombre}' no encontrado en DATASETS_CONFIG, saltando.")
            continue
        config = DATASETS_CONFIG[nombre]
        if not config["csv"].exists():
            logger.warning(f"  CSV no encontrado: {config['csv']}, saltando.")
            continue
        resultados[nombre] = medir_dataset(nombre, config)

    # --- Resumen ---
    logger.info("\n" + "="*60)
    logger.info("RESUMEN — Tiempos de Entrenamiento")
    logger.info("="*60)
    logger.info(f"  {'Dataset':<15} {'Train samples':>15} {'Features':>10} {'Train time':>12}")
    logger.info(f"  {'-'*55}")
    for nombre, r in resultados.items():
        train_str = f"{r['train_s']:.0f}s ({r['train_min']:.2f} min)"
        logger.info(f"  {nombre:<15} {r['n_train']:>15,} {r['n_features']:>10} {train_str:>12}")

    # --- Guardar JSON ---
    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados guardados en: {RESULTADOS_PATH}")

    # --- Líneas LaTeX listas para copiar ---
    logger.info("\n--- LaTeX (copiar en Tabla V) ---")
    for nombre, r in resultados.items():
        t = r['train_s']
        if t < 60:
            tiempo_tex = f"$\\approx$\\,{t:.0f}s"
        else:
            tiempo_tex = f"$\\approx$\\,{r['train_min']:.1f} min"
        logger.info(
            f"  {nombre:<15} & {r['n_train']:>15,} & {r['n_features']:>5} "
            f"& {tiempo_tex} \\\\"
        )


if __name__ == "__main__":
    main()
