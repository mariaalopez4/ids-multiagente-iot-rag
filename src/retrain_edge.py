"""
retrain_edge.py
===============
Reentrena el modelo XGBoost para Edge-IIoTset usando el pipeline actual
(preparar_datos_desde_csv), de forma que sea coherente con TON_IoT y Bot-IoT.

Ejecutar desde la raiz del proyecto:
    python src/retrain_edge.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

from agente_centinela import AgenteCentinela, preparar_datos_desde_csv
from config import DATASETS_CONFIG


def main():
    config = DATASETS_CONFIG["Edge-IIoT"]

    logger.info("Cargando Edge-IIoTset.csv ...")
    X, y = preparar_datos_desde_csv(
        str(config["csv"]),
        col_label="Attack_label",
        col_type="Attack_type"
    )
    logger.info(f"Datos listos: {len(X):,} filas, {X.shape[1]} features")
    logger.info(f"Clases ({y.nunique()}): {sorted(y.unique().tolist())}")

    logger.info("Entrenando XGBoost ...")
    centinela = AgenteCentinela()
    metricas = centinela.entrenar(X, y)

    logger.info("=== RESULTADOS ===")
    logger.info(f"Accuracy:  {metricas['accuracy']:.4f}")
    logger.info(f"F1-Macro:  {metricas['f1_macro']:.4f}")
    logger.info(f"Precision: {metricas['precision_macro']:.4f}")
    logger.info(f"Recall:    {metricas['recall_macro']:.4f}")
    logger.info(f"Clases:    {metricas['num_clases']}")
    logger.info(f"Train:     {metricas['num_train_samples']:,}")
    logger.info(f"Test:      {metricas['num_test_samples']:,}")

    ruta = str(config["modelo"])
    centinela.guardar_modelo(ruta)
    logger.info(f"Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
