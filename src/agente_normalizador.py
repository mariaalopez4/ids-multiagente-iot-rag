"""Agente Normalizador para mapeo semantico de features de red desconocidas."""

import os
import json
import math
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from groq import Groq
from config import MODELO_NORMALIZADOR

logger = logging.getLogger(__name__)


class _NanSafeEncoder(json.JSONEncoder):
    """Serializa np.nan como JSON null en lugar de NaN (invalido en JSON estandar)."""

    def _sanitize(self, obj):
        """Recorre recursivamente el objeto y sustituye float NaN por None."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)


class AgenteNormalizador:
    """
    Agente encargado de servir como 'Fallback'.
    Utiliza un LLM para mapear características de red desconocidas al esquema esperado por un modelo.
    """
    
    PROMPT_SISTEMA = """Eres un ingeniero experto en ciberseguridad y análisis de tráfico de red.
Tu tarea es mapear un log de tráfico de red (formato desconocido) a un conjunto de features canónicas
requeridas por un modelo XGBoost de detección de intrusiones IoT.

=== PROCESO DE MAPEO ===
1. Analiza semánticamente cada campo del log de entrada.
2. Para cada feature de destino, localiza el campo del input que mejor se corresponda
   (considera sinónimos, abreviaciones y equivalentes de protocolo).
3. Aplica las conversiones de tipo o unidad necesarias para obtener un valor numérico.

=== REGLAS DE PROTOCOLO ===
- Features de TCP (flags, seq, ack, syn, fin, rst): asigna null si el tráfico no es TCP.
- Features de UDP (stream, time_delta): asigna null si el tráfico no es UDP.
- Features de ICMP (checksum, seq): asigna null si el tráfico no es ICMP.
- Features de HTTP (código de respuesta, tamaño de cuerpo): asigna null si el servicio no es HTTP.
- Features de DNS (tipo de consulta): asigna null si el servicio no es DNS.

=== REGLAS GENERALES ===
1. Usa null (JSON null) para valores desconocidos o no aplicables.
   NUNCA uses 0.0 como sustituto de "ausente": el modelo distingue null de cero.
2. El modelo XGBoost es sparsity-aware: los null se enrutan por la rama "missing"
   del árbol de decisión, lo cual es el comportamiento correcto.
3. Responde ÚNICAMENTE con un objeto JSON válido que contenga exactamente
   las features indicadas. Sin texto adicional.
"""

    PROMPT_INFERENCIA_ESQUEMA = """Eres un experto en análisis de datasets de red.
Dado un ejemplo de las primeras filas de un dataset CSV (en formato JSON), tu tarea es identificar cuáles son las columnas clave para el entrenamiento del IDS.

Debes identificar:
1. "col_label": La columna que indica si el tráfico es Malicioso o Normal (suele ser binaria: 0/1 o strings como 'Normal'/'Attack').
2. "col_type": La columna que indica el tipo de ataque específico (ej: 'DDoS', 'Scan', 'SQLi'). Si solo hay una columna de etiqueta, usa la misma para ambas.

Reglas:
- Responde UNICAMENTE con un objeto JSON con las llaves "col_label" y "col_type".
- Si no estás seguro, elige las que más se parezcan semánticamente.
"""

    def __init__(self, api_key: str = None, modelo: str = MODELO_NORMALIZADOR):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere API Key de Groq para AgenteNormalizador")
        
        self.modelo = modelo
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info("AgenteNormalizador conectado a Groq exitosamente.")
        except Exception as e:
            logger.error(f"Fallo al conectar AgenteNormalizador con Groq: {e}")
            raise

    def _limpiar_json(self, texto: str) -> str:
        """Elimina posibles bloques de markdown o texto extra alrededor del JSON."""
        texto = texto.strip()
        if texto.startswith("```json"):
            texto = texto.split("```json")[1].split("```")[0].strip()
        elif texto.startswith("```"):
            texto = texto.split("```")[1].split("```")[0].strip()
        return texto

    def normalizar_features(self, log_entrada: Dict[str, Any], features_esperadas: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Mapea el log_entrada hacia el formato de features_esperadas.
        Retorna:
            - Dict mapeado (features_esperadas como llaves, flotantes como valores).
            - Tiempo que tardó el LLM en hacer el mapeo (latencia extra que asume el parche).
        """
        inicio = time.time()

        # Serialización segura: convierte listas, booleanos y NaN a tipos JSON válidos
        def _serializar_valor(v: Any) -> Any:
            if isinstance(v, bool):
                return int(v)  # True→1, False→0
            if isinstance(v, list):
                return str(v) if v else None  # lista vacía → null
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        log_serializable = {k: _serializar_valor(v) for k, v in log_entrada.items()}

        prompt_usuario = f"""INPUT LOG (formato JSON):
{json.dumps(log_serializable, indent=2, ensure_ascii=False)}

TARGET FEATURES EXPECTED:
{json.dumps(features_esperadas, ensure_ascii=False)}

Devuelve el JSON con exactamente estas {len(features_esperadas)} claves:"""
        
        try:
            respuesta = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.PROMPT_SISTEMA},
                    {"role": "user", "content": prompt_usuario}
                ],
                model=self.modelo,
                temperature=0.0, 
                response_format={"type": "json_object"}
            )
            
            contenido_raw = respuesta.choices[0].message.content
            # Limpiamos y parseamos
            contenido_json_str = self._limpiar_json(contenido_raw)
            mapeo_resultado = json.loads(contenido_json_str)
            
            # Post-procesamiento: null del LLM  →  np.nan (Sparsity-aware XGBoost)
            mapeo_seguro: Dict[str, Optional[float]] = {}
            for feature in features_esperadas:
                val = mapeo_resultado.get(feature, None)
                if val is None:
                    mapeo_seguro[feature] = np.nan
                else:
                    try:
                        mapeo_seguro[feature] = float(val)
                    except (ValueError, TypeError):
                        mapeo_seguro[feature] = np.nan
            
            # Cálculo de respaldo para avg_packet_size si es NaN pero tenemos los componentes
            if "avg_packet_size" in mapeo_seguro and np.isnan(mapeo_seguro["avg_packet_size"]):
                b = mapeo_seguro.get("bytes_src", np.nan)
                p = mapeo_seguro.get("pkts_src", np.nan)
                if not np.isnan(b) and not np.isnan(p) and p > 0:
                    mapeo_seguro["avg_packet_size"] = b / p

            tiempo_total = time.time() - inicio
            logger.info(f"Normalizador LLM completó el mapeo en {tiempo_total:.2f}s")
            
            return mapeo_seguro, tiempo_total
            
        except Exception as e:
            tiempo_total = time.time() - inicio
            logger.error(f"Error en el LLM de normalización, cayendo a NaN para todos los campos. Error: {e}")

            mapeo_seguro = {f: np.nan for f in features_esperadas}
            return mapeo_seguro, tiempo_total

    def inferir_esquema_dataset(self, df_sample: Any) -> Dict[str, str]:
        """
        Analiza una muestra del dataset (5-10 filas) para identificar
        semánticamente cuáles son las columnas de etiqueta (Label) y tipo (Type).
        """
        logger.info("Iniciando inferencia semántica del esquema del dataset...")
        
        # Convertimos el sample a un formato que el LLM entienda bien
        if hasattr(df_sample, 'to_dict'):
            sample_data = df_sample.to_dict(orient='records')
        else:
            sample_data = df_sample

        prompt_usuario = f"""
        MUESTRA DEL DATASET (Primeras filas):
        {json.dumps(sample_data, indent=2)}
        
        Identifica 'col_label' y 'col_type'.
        """

        try:
            respuesta = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.PROMPT_INFERENCIA_ESQUEMA},
                    {"role": "user", "content": prompt_usuario}
                ],
                model=self.modelo,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            contenido_raw = respuesta.choices[0].message.content
            esquema_str = self._limpiar_json(contenido_raw)
            esquema = json.loads(esquema_str)
            logger.info(f"Esquema inferido exitosamente: {esquema}")
            return esquema
            
        except Exception as e:
            logger.error(f"Fallo al inferir esquema con LLM: {e}. Usando defaults de Edge-IIoT.")
            return {"col_label": "Attack_label", "col_type": "Attack_type"}

if __name__ == "__main__":
    # Test rápido de ejecución local
    print("Agente Normalizador - Módulo preparado")
