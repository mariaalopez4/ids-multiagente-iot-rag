"""Agente Centinela para deteccion de ataques IoT mediante XGBoost."""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config import LEAKAGE_INDICATORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)



class AgenteCentinela:
    # Clases del modelo experto Edge-IIoT (14 ataques + tráfico normal).
    # Solo se usa en generar_metricas.py para filtrar el classification_report.
    ATAQUES_CONOCIDOS = [
        'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
        'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
        'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
    ]
    
    def __init__(self, modelo_path: Optional[str] = None):
        self.modelo = None
        self.label_encoder = LabelEncoder()
        self.clases = {}
        self.metricas = {}
        self.esta_entrenado = False
        
        if modelo_path and Path(modelo_path).exists():
            self.cargar_modelo(modelo_path)
    
    def entrenar(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 50,
        verbose: bool = True
    ) -> Dict:
        """Entrena el clasificador XGBoost y retorna métricas de evaluación."""
        if X.empty or y.empty:
            raise ValueError("Los datos de entrenamiento no pueden estar vacíos")
        
        if len(X) != len(y):
            raise ValueError(f"X tiene {len(X)} filas pero y tiene {len(y)}")
        
        logger.info(f"Iniciando entrenamiento: {len(X)} muestras, {X.shape[1]} features")
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.clases = dict(enumerate(self.label_encoder.classes_))
        logger.info(f"Clases detectadas: {len(self.clases)}")
        
        X = X.copy()
        X.columns = X.columns.str.replace('[', '(', regex=False)
        X.columns = X.columns.str.replace(']', ')', regex=False)
        X.columns = X.columns.str.replace('<', 'lt', regex=False)
        X.columns = X.columns.str.replace('>', 'gt', regex=False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        if len(self.clases) == 2:
            self.modelo = XGBClassifier(
                objective='binary:logistic',
                n_estimators=n_estimators,
                eval_metric='logloss',
                random_state=random_state,
                verbosity=1 if verbose else 0
            )
        else:
            self.modelo = XGBClassifier(
                objective='multi:softprob',
                num_class=len(self.clases),
                n_estimators=n_estimators,
                eval_metric='mlogloss',
                random_state=random_state,
                verbosity=1 if verbose else 0
            )
        
        logger.info("Entrenando modelo...")
        self.modelo.fit(X_train, y_train)
        
        y_pred = self.modelo.predict(X_test)
        
        self.metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'num_features': X.shape[1],
            'num_clases': len(self.clases),
            'fecha_entrenamiento': datetime.now().isoformat()
        }
        
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        self.metricas['reporte_clases'] = report
        
        cm = confusion_matrix(y_test, y_pred)
        self.metricas['confusion_matrix'] = cm.tolist()
        
        self.esta_entrenado = True
        
        logger.info("Entrenamiento completado")
        logger.info(f"Accuracy: {self.metricas['accuracy']*100:.2f}%")
        logger.info(f"F1-Score (macro): {self.metricas['f1_macro']:.4f}")
        
        return self.metricas
    
    def predecir(self, features: np.ndarray) -> Tuple[str, float]:
        if not self.esta_entrenado:
            raise RuntimeError(
                "El modelo no está entrenado. Ejecuta .entrenar() o .cargar_modelo()"
            )
        
        if features is None or (isinstance(features, np.ndarray) and features.size == 0):
            raise ValueError("Features no pueden estar vacías")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        expected_features = self.modelo.n_features_in_
        if features.shape[1] != expected_features:
            raise ValueError(
                f"Se esperan {expected_features} features, "
                f"pero se recibieron {features.shape[1]}"
            )
        
        prediccion_num = self.modelo.predict(features)[0]
        probabilidades = self.modelo.predict_proba(features)[0]
        
        nombre_ataque = self.label_encoder.inverse_transform([prediccion_num])[0]
        confianza = float(probabilidades[prediccion_num])
        
        logger.debug(f"Predicción: {nombre_ataque} (confianza: {confianza:.2%})")
        
        return nombre_ataque, confianza
    
    def predecir_batch(self, features_batch: np.ndarray) -> List[Tuple[str, float]]:
        if not self.esta_entrenado:
            raise RuntimeError("Modelo no entrenado")
        
        predicciones_num = self.modelo.predict(features_batch)
        probabilidades = self.modelo.predict_proba(features_batch)
        
        resultados = []
        for i, pred_num in enumerate(predicciones_num):
            nombre = self.label_encoder.inverse_transform([pred_num])[0]
            confianza = float(probabilidades[i][pred_num])
            resultados.append((nombre, confianza))
        
        return resultados
    
    def guardar_modelo(self, ruta: str):
        if not self.esta_entrenado:
            raise RuntimeError("No hay modelo entrenado para guardar")
        
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        
        datos_modelo = {
            'modelo': self.modelo,
            'label_encoder': self.label_encoder,
            'clases': self.clases,
            'metricas': self.metricas,
            'version': '1.0',
            'fecha_guardado': datetime.now().isoformat()
        }
        
        joblib.dump(datos_modelo, ruta)
        logger.info(f"Modelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta: str):
        if not Path(ruta).exists():
            raise FileNotFoundError(f"No se encuentra el modelo en: {ruta}")
        
        datos_modelo = joblib.load(ruta)
        
        self.modelo = datos_modelo['modelo']
        self.label_encoder = datos_modelo['label_encoder']
        self.clases = datos_modelo['clases']
        self.metricas = datos_modelo.get('metricas', {})
        self.esta_entrenado = True
        
        logger.info(f"Modelo cargado desde: {ruta}")
        logger.info(f"Entrenado el: {self.metricas.get('fecha_entrenamiento', 'N/A')}")
    
    def exportar_metricas(self, ruta: str):
        if not self.metricas:
            raise RuntimeError("No hay métricas disponibles. Entrena el modelo primero.")
        
        metricas_export = self.metricas.copy()
        
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(metricas_export, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas exportadas a: {ruta}")
    
    def obtener_importancia_features(self, top_n: int = 20) -> Dict[str, float]:
        if not self.esta_entrenado:
            raise RuntimeError("Modelo no entrenado")
        
        importancias = self.modelo.feature_importances_
        nombres_features = self.modelo.get_booster().feature_names
        
        feature_dict = dict(zip(nombres_features, importancias))
        feature_dict_sorted = dict(
            sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return feature_dict_sorted
        
    @property
    def features_esperadas(self) -> List[str]:
        """Devuelve la lista exacta de nombres de features que el modelo espera recibir."""
        if not self.esta_entrenado or not self.modelo:
            return []
        try:
            return self.modelo.get_booster().feature_names
        except Exception:
            # Fallback en caso de que xgboost cambie su API
            if hasattr(self.modelo, 'feature_names_in_'):
                return list(self.modelo.feature_names_in_)
            return []
    
    def resumen(self) -> str:
        """Genera un resumen textual del estado del modelo."""
        if not self.esta_entrenado:
            return "Agente Centinela NO entrenado. Ejecuta .entrenar() primero."
        
        resumen = f"""
AGENTE CENTINELA - ESTADO DEL SISTEMA

Metricas de Rendimiento:
   • Accuracy: {self.metricas.get('accuracy', 0)*100:.2f}%
   • F1-Score (Macro): {self.metricas.get('f1_macro', 0):.4f}
   • Precision (Macro): {self.metricas.get('precision_macro', 0):.4f}
   • Recall (Macro): {self.metricas.get('recall_macro', 0):.4f}

Configuracion:
   • Clases detectadas: {self.metricas.get('num_clases', 0)}
   • Features utilizadas: {self.metricas.get('num_features', 0)}
   • Muestras de entrenamiento: {self.metricas.get('num_train_samples', 0):,}
   • Muestras de test: {self.metricas.get('num_test_samples', 0):,}

Última actualización: {self.metricas.get('fecha_entrenamiento', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return resumen.strip()


def preparar_datos_desde_csv(
    ruta_csv: str, 
    col_label: str = 'Attack_label', 
    col_type: str = 'Attack_type'
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Cargando datos desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv, low_memory=False)
    
    # Manejar posibles columnas faltantes informando al usuario
    cols_actuales = df.columns.tolist()
    if col_label not in cols_actuales or col_type not in cols_actuales:
        logger.warning(f"Columnas {col_label}/{col_type} no encontradas en {ruta_csv}. Reintentando con autodetect...")
        # Fallback simple: buscar columnas que contengan 'label' o 'type' o 'attack'
        if 'label' in cols_actuales: col_label = 'label'
        if 'type' in cols_actuales: col_type = 'type'
        elif 'attack' in cols_actuales: col_type = 'attack'

    # 1. Guardar y (etiqueta) ANTES de cualquier transformacion numerica.
    #    La columna de texto se convertiria en NaN si la coecionamos despues.
    y = df[col_type]

    # 2. Limpieza de Data Leakage: columnas que 'darian la respuesta' al modelo.
    columnas_a_borrar = set([col_label, col_type])
    for c in list(df.columns):
        if any(p in c.lower() for p in LEAKAGE_INDICATORS) and c not in [col_label, col_type]:
            columnas_a_borrar.add(c)

    # 3. Coercion numerica universal.
    #    Edge-IIoTset tiene columnas dummy (0/1) con dtype bool que
    #    select_dtypes excluye (bool != np.number). Las convertimos
    #    a float. Las columnas object con texto puro se convierten en NaN.
    df_work = df.drop(columns=list(columnas_a_borrar), errors='ignore')
    for col in df_work.columns:
        if df_work[col].dtype == bool:
            df_work[col] = df_work[col].astype(float)
        elif df_work[col].dtype == object:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

    # 4. Seleccionar solo columnas numericas y eliminar las todo-NaN.
    X = df_work.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how='all')

    logger.info(f"Datos cargados: {len(df)} registros. Features: {X.shape[1]}")
    return X, y


if __name__ == "__main__":
    print("Agente Centinela - Modulo cargado correctamente")
