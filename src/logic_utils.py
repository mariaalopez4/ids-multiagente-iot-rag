"""Utilidades de logica de negocio para la interfaz Streamlit."""

import pandas as pd
import json
from config import LEAKAGE_INDICATORS

def limpiar_fuga_datos(muestra: pd.Series, col_label: str, col_type: str) -> dict:
    """
    Elimina columnas que puedan causar Data Leakage antes de pasar los datos al modelo.
    """
    columnas_a_quitar = [col_label, col_type]
    for c in muestra.index:
        if any(p in str(c).lower() for p in LEAKAGE_INDICATORS):
            columnas_a_quitar.append(c)
    
    features_dict = muestra.drop(columnas_a_quitar, errors='ignore').to_dict()
    return features_dict

def format_log_preview(features_dict: dict, max_items: int = 25) -> str:
    """
    Formatea un diccionario de features para una vista previa de log elegante.
    """
    items_log = list(features_dict.items())
    log_preview = " | ".join([f"{k}: {v}" for k, v in items_log[:max_items]])
    if len(items_log) > max_items:
        log_preview += " | ..."
    return log_preview

def validar_esquema_minimo(json_data: dict, min_features: int = 3) -> bool:
    """
    Valida que el JSON tenga un número mínimo de claves para ser procesable.
    """
    if not isinstance(json_data, dict):
        return False
    return len(json_data) >= min_features
