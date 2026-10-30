"""Sistema multiagente de ciberseguridad IoT.

Coordina deteccion, analisis y respuesta de ataques.
Incluye Alta Disponibilidad: Fallback LLM + Entrenamiento en Background.
"""

import logging
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from agente_centinela import AgenteCentinela, preparar_datos_desde_csv
from agente_analista import AgenteAnalista
from agente_respondedor import AgenteRespondedor
from agente_normalizador import AgenteNormalizador

from config import (
    MITRE_KB, DATASETS_CONFIG, DATASET_DEFAULT, 
    RESULTADOS_DIR, FAISS_INDEX_DIR, LEARNED_SCHEMAS_PATH,
    MODELO_FALLBACK, FEATURES_FALLBACK, AUTODETECCION_ID
)


# Configuración manual del Logger para evitar conflictos con Streamlit
logger = logging.getLogger("Orquestador")
logger.setLevel(logging.INFO)
logger.propagate = False  # Evita que Streamlit duplique los mensajes en la consola

# Limpiamos handlers previos por si Streamlit recarga el módulo
if logger.hasHandlers():
    logger.handlers.clear()

# Creamos un formateador con la fecha en estilo europeo/español (Día/Mes/Año)
formato_log = logging.Formatter(
    fmt='%(asctime)s - [ORQUESTADOR] - %(levelname)s - %(message)s', 
    datefmt='%d/%m/%Y %H:%M:%S'
)

# Handler para escribir en el archivo de texto
file_handler = logging.FileHandler(RESULTADOS_DIR / "sistema_ciberseguridad.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formato_log)
logger.addHandler(file_handler)

# Handler para imprimir en la consola
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formato_log)
logger.addHandler(stream_handler)

# --- Ruta del archivo de estado compartido con Streamlit ---
# Este archivo actúa como "mensajero" entre el hilo de background
# y la interfaz gráfica, ya que Streamlit recarga el script entero
# en cada interacción y perdería cualquier variable en memoria del hilo.
ESTADO_ENTRENAMIENTO_PATH = RESULTADOS_DIR / "estado_entrenamiento.json"


def _leer_estado_entrenamiento() -> Dict:
    """Lee el archivo JSON de estado del entrenamiento desde disco."""
    if ESTADO_ENTRENAMIENTO_PATH.exists():
        try:
            with open(ESTADO_ENTRENAMIENTO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _escribir_estado_entrenamiento(dataset_id: str, estado: str):
    """Escribe el estado de un dataset concreto en el archivo JSON de estado."""
    datos = _leer_estado_entrenamiento()
    datos[dataset_id] = estado
    ESTADO_ENTRENAMIENTO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ESTADO_ENTRENAMIENTO_PATH, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2)
    logger.info(f"[Estado] {dataset_id} -> '{estado}' guardado en {ESTADO_ENTRENAMIENTO_PATH}")


class SistemaCiberseguridad:
    def __init__(
        self,
        api_key_groq: str,
        modo_simulacion: bool = True,
        # Por retrocompatibilidad: si se pasa el path de un modelo, se carga como default
        modelo_centinela_path: Optional[str] = None,
    ):
        """
        Inicializa el sistema integrando todos los agentes.
        """
        self.modo_simulacion = modo_simulacion
        self.api_key_groq = api_key_groq
        
        # --- Control de concurrencia ---
        # Este Lock garantiza que el hilo de background que entrena
        # un modelo nuevo no choca con el hilo principal que está haciendo predicciones.
        self._lock = threading.Lock()
        
        # --- Registro de hilos activos ---
        # Set para saber qué datasets están siendo entrenados ahora mismo
        # y evitar lanzar el mismo entrenamiento dos veces.
        self._entrenando = set()

        logger.info("Inicializando Sistema Multiagente de Defensa IoT (Alta Disponibilidad)...")

        # --- Datasets y Esquemas Aprendidos ---
        self.centinelas: Dict[str, AgenteCentinela] = {}
        self.learned_schemas: Dict[str, Dict[str, str]] = self._cargar_schemas_aprendidos()
        
        self._cargar_modelos_disponibles()
        
        # Por retrocompatibilidad: si se pasó un modelo concreto y no cargó el default
        if modelo_centinela_path and DATASET_DEFAULT not in self.centinelas:
            try:
                c = AgenteCentinela(modelo_path=modelo_centinela_path)
                if c.esta_entrenado:
                    self.centinelas[DATASET_DEFAULT] = c
                    logger.info(f"Centinela base cargado desde argumento: {modelo_centinela_path}")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo del argumento: {e}")

        # --- Agente base (retrocompatibilidad) ---
        # Para que el código existente que accede a 'sistema.centinela' siga funcionando
        self.centinela = self.centinelas.get(DATASET_DEFAULT)

        # --- Modelo de Emergencia (Fallback Universal) ---
        self.centinela_fallback = None
        if MODELO_FALLBACK.exists():
            try:
                self.centinela_fallback = AgenteCentinela(modelo_path=str(MODELO_FALLBACK))
                logger.info("Agente Fallback (Mecánico de Emergencias): LISTO")
            except Exception as e:
                logger.error(f"No se pudo cargar el modelo de emergencia Fallback: {e}")

        # --- Agente Analista (LLM) ---
        try:
            self.analista = AgenteAnalista(api_key=api_key_groq)
            logger.info("Agente Analista: LISTO")
        except Exception as e:
            logger.error(f"Fallo al iniciar Analista: {e}")
            raise

        # --- Agente Respondedor (RAG) ---
        try:
            self.respondedor = AgenteRespondedor(
                api_key=api_key_groq,
                ruta_kb=str(MITRE_KB),
                ruta_indices=str(FAISS_INDEX_DIR) # Fijamos la ruta para evitar errores de escritura
            )
            logger.info("Agente Respondedor: LISTO (Índice RAG cargado)")
        except Exception as e:
            logger.error(f"Fallo al iniciar Respondedor: {e}")
            raise

        # --- Agente Normalizador (Fallback LLM) ---
        try:
            self.normalizador = AgenteNormalizador(api_key=api_key_groq)
            logger.info("Agente Normalizador: LISTO (Fallback LLM activo)")
        except Exception as e:
            logger.error(f"Fallo al iniciar Normalizador: {e}")
            raise

        logger.info("SISTEMA COMPLETAMENTE OPERATIVO\n")

    def registrar_nuevo_centinela(self, dataset_id: str, centinela: AgenteCentinela):
        """Registra un nuevo agente centinela en el sistema en caliente."""
        with self._lock:
            self.centinelas[dataset_id] = centinela
            _escribir_estado_entrenamiento(dataset_id, "listo")
        logger.info(f"Nuevo agente experto registrado para: {dataset_id}")


    # ------------------------------------------------------------------ #
    #  MÉTODOS PRIVADOS DE GESTIÓN DE MODELOS                             #
    # ------------------------------------------------------------------ #

    def _cargar_modelos_disponibles(self):
        """
        Recorre el DATASETS_CONFIG y carga en memoria todos los modelos
        cuyo archivo .pkl ya existe en disco.
        """
        for dataset_id, config in DATASETS_CONFIG.items():
            ruta_modelo = config.get("modelo")
            if not ruta_modelo:
                continue  # Entradas sin modelo (ej: Autodetección) se saltan
            if ruta_modelo.exists():
                try:
                    centinela = AgenteCentinela(modelo_path=str(ruta_modelo))
                    if centinela.esta_entrenado:
                        self.centinelas[dataset_id] = centinela
                        logger.info(f"Centinela cargado: '{dataset_id}' ({ruta_modelo.name})")
                except Exception as e:
                    logger.warning(f"No se pudo cargar modelo para '{dataset_id}': {e}")
            else:
                logger.info(f"Sin modelo para '{dataset_id}' (se usará Fallback LLM si se solicita)")

    def _entrenar_en_background(self, dataset_id: str):
        """
        Hilo de background: entrena un XGBoost para el dataset_id dado.
        
        Flujo:
        1. Marca el estado como "entrenando" en el JSON de estado.
        2. Carga el CSV y entrena el modelo XGBoost.
        3. Guarda el modelo en disco.
        4. Con el Lock adquirido, actualiza el diccionario en memoria.
        5. Marca el estado como "listo".
        """
        config = DATASETS_CONFIG.get(dataset_id)
        if not config:
            logger.error(f"[BACKGROUND] Dataset '{dataset_id}' no existe en DATASETS_CONFIG.")
            return

        ruta_csv = config["csv"]
        ruta_modelo = config["modelo"]

        logger.info(f"[BACKGROUND] Iniciando entrenamiento para '{dataset_id}'...")
        _escribir_estado_entrenamiento(dataset_id, "entrenando")

        try:
            # 1. Obtener o Inferir Esquema Semántico
            col_label = config.get("col_label")
            col_type = config.get("col_type")
            
            if not col_label or not col_type:
                # Si no está en config, miramos en caché
                schema = self.learned_schemas.get(dataset_id)
                if schema:
                    col_label, col_type = schema["col_label"], schema["col_type"]
                else:
                    # Fase Inferencia: Preguntamos al LLM analizando una muestra
                    logger.info(f"[BACKGROUND] Analizando estructura semántica de '{dataset_id}'...")
                    sample_df = pd.read_csv(ruta_csv, nrows=5)
                    schema = self.normalizador.inferir_esquema_dataset(sample_df)
                    col_label, col_type = schema["col_label"], schema["col_type"]
                    self._guardar_schema_aprendido(dataset_id, col_label, col_type)

            # 2. Cargar datos usando el esquema aprendido
            logger.info(f"[BACKGROUND] Cargando datos con esquema: {col_label}/{col_type}")
            X, y = preparar_datos_desde_csv(
                str(ruta_csv), 
                col_label=col_label, 
                col_type=col_type
            )

            # 3. Entrenar modelo
            nuevo_centinela = AgenteCentinela()
            nuevo_centinela.entrenar(X, y)

            # 4. Guardar en disco
            nuevo_centinela.guardar_modelo(str(ruta_modelo))
            logger.info(f"[BACKGROUND] Modelo guardado en: {ruta_modelo}")

            # 5. Actualizar en memoria con el candado (threading.Lock)
            with self._lock:
                self.centinelas[dataset_id] = nuevo_centinela
                if dataset_id == DATASET_DEFAULT:
                    self.centinela = nuevo_centinela  # retrocompatibilidad

            # 6. Marcar como listo para que el Hot-Swap sea inmediato
            _escribir_estado_entrenamiento(dataset_id, "listo")
            logger.info(
                f"[BACKGROUND] Entrenamiento completado para '{dataset_id}'. "
                "Hot-Swap al modo Experto (Fase 3) realizado con éxito."
            )

            # 7. Lanzar la evaluación comparativa en un hilo separado (no bloquea)
            eval_thread = threading.Thread(
                target=self._evaluar_metricas_background,
                args=(dataset_id,),
                daemon=True
            )
            eval_thread.start()

        except Exception as e:
            logger.error(f"[BACKGROUND] Error entrenando '{dataset_id}': {e}")
            _escribir_estado_entrenamiento(dataset_id, "error")
        finally:
            # Eliminar del registro de entrenamientos activos
            self._entrenando.discard(dataset_id)

    def _evaluar_metricas_background(self, dataset_id: str):
        """
        Lanza el evaluador comparativo (Fase 1 vs Fase 3) en segundo plano
        tras completar el entrenamiento de un modelo nativo.
        Usa 5 muestras para no agotar los créditos de la API Groq.
        """
        try:
            from evaluador_comparativo import EvaluadorComparativo
            logger.info(f"[EVAL METRICS] Iniciando evaluación comparativa para '{dataset_id}'...")
            evaluador = EvaluadorComparativo(api_key_groq=self.api_key_groq, n_muestras=5)
            evaluador.evaluar(dataset_id)
        except ImportError:
            logger.info("[EVAL METRICS] Módulo evaluador_comparativo no disponible. Omitiendo.")
        except Exception as e:
            logger.warning(f"[EVAL METRICS] No se pudo completar la evaluación comparativa: {e}")

    # ------------------------------------------------------------------ #
    #  MEMORIA PERSISTENTE DE ESQUEMAS                                     #
    # ------------------------------------------------------------------ #

    def _cargar_schemas_aprendidos(self) -> Dict:
        """Carga la memoria de esquemas identificados por el LLM en sesiones previas."""
        if LEARNED_SCHEMAS_PATH.exists():
            try:
                with open(LEARNED_SCHEMAS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _guardar_schema_aprendido(self, dataset_id: str, col_label: str, col_type: str):
        """Guarda un nuevo esquema inferido en la memoria persistente de disco."""
        self.learned_schemas[dataset_id] = {
            "col_label": col_label,
            "col_type": col_type,
            "fecha_aprendizaje": datetime.now().isoformat()
        }
        try:
            with open(LEARNED_SCHEMAS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.learned_schemas, f, indent=2)
            logger.info(f"Memoria de esquemas actualizada para '{dataset_id}'")
        except Exception as e:
            logger.error(f"No se pudo guardar la memoria de esquemas: {e}")

    def _lanzar_entrenamiento_si_posible(self, dataset_id: str):
        """
        Comprueba si se puede arrancar el entrenamiento en background:
        - Que no esté ya entrenándose.
        - Que el CSV exista en disco.
        Si ambas condiciones se cumplen, lanza el hilo.
        """
        if dataset_id in self._entrenando:
            logger.info(f"Entrenamiento de '{dataset_id}' ya está en curso. Esperando.")
            return

        config = DATASETS_CONFIG.get(dataset_id)
        if not config:
            logger.warning(f"No hay configuración para dataset '{dataset_id}'.")
            return

        ruta_csv = config["csv"]
        if not ruta_csv.exists():
            logger.warning(
                f"CSV no encontrado para '{dataset_id}' en {ruta_csv}. "
                "El sistema permanecerá en modo Fallback LLM hasta que se proporcione el dataset."
            )
            _escribir_estado_entrenamiento(dataset_id, "sin_dataset")
            return

        # Todo listo: registrar y lanzar el hilo
        self._entrenando.add(dataset_id)
        hilo = threading.Thread(
            target=self._entrenar_en_background,
            args=(dataset_id,),
            daemon=True  # daemon=True: el hilo se destruye si se cierra el programa principal
        )
        hilo.start()
        logger.info(f"Hilo de entrenamiento lanzado para '{dataset_id}'.")

    def obtener_columnas_esquema(self, dataset_id: str) -> Dict[str, str]:
        """
        Método público para la UI: devuelve col_label y col_type.
        Si no se conocen, dispara la inferencia semántica inmediatamente.
        """
        config = DATASETS_CONFIG.get(dataset_id, {})
        col_label = config.get("col_label")
        col_type = config.get("col_type")

        if col_label and col_type:
            return {"col_label": col_label, "col_type": col_type}
        
        # Consultar memoria aprendida
        schema = self.learned_schemas.get(dataset_id)
        if schema:
            return {"col_label": schema["col_label"], "col_type": schema["col_type"]}
        
        # Si no lo sabemos, inferimos ahora mismo (bloqueante para la UI pero necesario)
        ruta_csv = config.get("csv")
        if ruta_csv and ruta_csv.exists():
            logger.info(f"Inferiendo esquema para la UI: {dataset_id}")
            sample_df = pd.read_csv(ruta_csv, nrows=5)
            schema = self.normalizador.inferir_esquema_dataset(sample_df)
            self._guardar_schema_aprendido(dataset_id, schema["col_label"], schema["col_type"])
            return schema
            
        return {"col_label": "Attack_label", "col_type": "Attack_type"} # Default fallback

    # ------------------------------------------------------------------ #
    #  PIPELINE PRINCIPAL                                                  #
    # ------------------------------------------------------------------ #

    def procesar_trafico(self, log_trafico: Dict, dataset_id: str = DATASET_DEFAULT) -> Dict:
        """
        Ejecuta el pipeline completo: detección -> análisis -> respuesta.
        
        Nuevos parámetros:
        - dataset_id: el origen del tráfico (ej: "TON_IoT"). Determina qué modelo usar.
        
        Nuevos campos en el resultado:
        - fallback_activo: True si se usó el Normalizador LLM en lugar del modelo específico.
        - tiempo_normalizacion_llm: segundos que tardó el LLM en mapear features (si aplica).
        """
        inicio_proceso = time.time()
        id_evento = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        resultado_final = {
            "id_evento": id_evento,
            "timestamp": datetime.now().isoformat(),
            "dataset_origen": dataset_id,
            "log_original": log_trafico,
            "deteccion": None,
            "analisis": None,
            "respuesta": None,
            "acciones_tomadas": [],
            "fallback_activo": False,
            "tiempo_normalizacion_llm": 0.0,
        }

        logger.info(f"Procesando evento ID: {id_evento} | Dataset: {dataset_id}")

        try:
            if not log_trafico:
                raise ValueError("log_trafico no puede estar vacío")

            # --- Adquirir el candado solo para LEER el modelo activo ---
            # Es una lectura muy rápida, así no bloqueamos el sistema.
            with self._lock:
                centinela_activo = self.centinelas.get(dataset_id)
                centinela_base = self.centinelas.get(DATASET_DEFAULT)

            # Sanitización previa: Filtrar solo lo numérico para evitar crashes
            log_sanitizado = {}
            for k, v in log_trafico.items():
                try:
                    val = float(v)
                    if np.isfinite(val):
                        log_sanitizado[k] = val
                except (TypeError, ValueError):
                    continue

            # Cuando el dataset_id es AUTODETECCION_ID, el sistema no intenta buscar
            # ningún modelo experto. Activa directamente el Normalizador LLM,
            # lo que permite analizar cualquier log de red desconocido sin configuración previa.
            if dataset_id == AUTODETECCION_ID:
                if not self.centinela_fallback:
                    raise RuntimeError(
                        "Modo Autodetección activo pero el Modelo Fallback Universal no está disponible. "
                        "Asegúrate de que 'xgboost_fallback_model.pkl' existe en la carpeta 'modelos/'."
                    )
                logger.info(
                    "[AUTODETECCIÓN] Contexto de red desconocido. "
                    "Activando Normalizador LLM + Fallback Universal directamente..."
                )
                resultado_final["fallback_activo"] = True
                features_mapeadas, tiempo_llm = self.normalizador.normalizar_features(
                    log_entrada=log_trafico,
                    features_esperadas=FEATURES_FALLBACK
                )
                resultado_final["tiempo_normalizacion_llm"] = tiempo_llm
                ruta_mapeo = RESULTADOS_DIR / "ultimo_mapeo_llm.json"
                try:
                    from agente_normalizador import _NanSafeEncoder
                    with open(ruta_mapeo, "w", encoding="utf-8") as f:
                        json.dump(features_mapeadas, f, indent=2, cls=_NanSafeEncoder)
                except Exception:
                    pass
                features = np.array(list(features_mapeadas.values()), dtype=float).reshape(1, -1)
                centinela_activo = self.centinela_fallback
                modelo_usado = "LLM Normalizador + Fallback Universal (Autodetección)"

            else:
                # El dataset_id es un contexto conocido: buscar modelo experto o activar fallback
                if centinela_activo:
                    # CASO A: Modelo específico disponible -> Inferencia directa
                    # Nos aseguramos de pasar las columnas en el orden EXACTO que el modelo espera
                    features_lista = []
                    for f in centinela_activo.features_esperadas:
                        features_lista.append(log_sanitizado.get(f, 0.0))
                    
                    features = np.array(features_lista, dtype=float).reshape(1, -1)
                    if not np.all(np.isfinite(features)):
                        logger.warning("Features no finitas detectadas en inferencia directa. Aplicando saneamiento a 0.0.")
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    modelo_usado = dataset_id
                    tiempo_llm = 0.0

                elif self.centinela_fallback:
                    # CASO B: No hay modelo específico -> FALLBACK LLM
                    logger.warning(
                        f"No hay modelo para '{dataset_id}'. "
                        "Activando Agente de Fallback Universal (15 features)..."
                    )
                    resultado_final["fallback_activo"] = True

                    # Paso 1: Pedir al Normalizador LLM que mapee SÓLO las 15 universales
                    features_mapeadas, tiempo_llm = self.normalizador.normalizar_features(
                        log_entrada=log_trafico,
                        features_esperadas=FEATURES_FALLBACK
                    )
                    resultado_final["tiempo_normalizacion_llm"] = tiempo_llm
                    
                    # Exportar JSON de mapeo para auditoría
                    ruta_mapeo = RESULTADOS_DIR / "ultimo_mapeo_llm.json"
                    try:
                        from agente_normalizador import _NanSafeEncoder
                        with open(ruta_mapeo, "w", encoding="utf-8") as f:
                            json.dump(features_mapeadas, f, indent=2, cls=_NanSafeEncoder)
                        logger.info(f"Mapeo LLM guardado en {ruta_mapeo}")
                    except Exception:
                        pass

                    # Paso 2: Construir el vector preservando los np.nan (Sparsity-aware)
                    features = np.array(
                        list(features_mapeadas.values()), dtype=float
                    ).reshape(1, -1)
                    
                    # Preservamos los NaN orgánicos del LLM para que el XGBoost Fallback tome la rama adecuada.
                    
                    centinela_activo = self.centinela_fallback
                    modelo_usado = f"Fallback Universal ({len(FEATURES_FALLBACK)} common features)"

                    # Paso 3: Intentar arrancar entrenamiento en background de un modelo nativo
                    self._lanzar_entrenamiento_si_posible(dataset_id)
                elif centinela_base:
                    # CASO B.2: Solo usado si falla la carga del fallback pero existe el experto base
                    logger.error("Fallback Universal no cargado, intentando usar experto antiguo.")
                    centinela_activo = centinela_base
                    modelo_usado = f"{DATASET_DEFAULT} (LEGACY)"
                    self._lanzar_entrenamiento_si_posible(dataset_id)
                else:
                    # CASO C: No hay ni modelo base -> Sistema aún no inicializado
                    raise RuntimeError(
                        "No hay ningún modelo disponible (ni base ni específico). "
                        "Entrena el modelo base primero."
                    )

            # --- Detección ---
            tipo_ataque, confianza = centinela_activo.predecir(features)

            resultado_final["deteccion"] = {
                "tipo_ataque": tipo_ataque,
                "confianza": confianza,
                "modelo": modelo_usado
            }
            logger.info(f"Centinela detectó: {tipo_ataque} ({confianza:.2%}) | Modelo: {modelo_usado}")

        except Exception as e:
            logger.error(f"Error en Centinela: {e}")
            return resultado_final

        if str(tipo_ataque).lower() == "normal":
            logger.info("Tráfico legítimo. No se requieren más acciones.")
            resultado_final["tiempo_total_procesamiento"] = time.time() - inicio_proceso
            return resultado_final

        # --- Análisis LLM ---
        try:
            analisis = self.analista.analizar(tipo_ataque=tipo_ataque, contexto=log_trafico)
            resultado_final["analisis"] = {
                "explicacion": analisis.explicacion,
                "severidad": analisis.severidad,
                "tactica_mitre": analisis.tactica_mitre
            }
            logger.info(f"Analista reporta: Severidad {analisis.severidad}")
        except Exception as e:
            logger.error(f"Error en Analista: {e}")

        # --- Respuesta RAG (Contextual) ---
        try:
            # Proveer evidencia técnica abstraída como contexto analítico para mejorar la precisión del RAG
            evidencia = analisis.explicacion if analisis else "No disponible"
            
            # Si el tipo de ataque es genérico (Modo Supervivencia), usamos la deducción del Analista
            tipo_para_rag = tipo_ataque
            if tipo_ataque.lower() == "ataque" and analisis and hasattr(analisis, "tipo_ataque"):
                tipo_para_rag = analisis.tipo_ataque

            contexto_experto = f"{tipo_para_rag} (Evidencia técnica: {evidencia})"
            
            plan_respuesta = self.respondedor.consultar(contexto_experto)
            resultado_final["respuesta"] = {
                "mitigacion_tecnica": plan_respuesta['mitigacion'],
                "fuentes_cap": [doc.metadata.get('source', 'unknown') for doc in plan_respuesta['fuentes']]
            }
            logger.info("Respuesta RAG generada correctamente")
        except Exception as e:
            logger.error(f"Error en Respondedor: {e}")

        if self.modo_simulacion:
            accion = "[SIMULACIÓN] Se aplicaría regla de firewall para bloquear origen."
            resultado_final["acciones_tomadas"].append(accion)
            logger.info(accion)

        resultado_final["tiempo_total_procesamiento"] = time.time() - inicio_proceso
        return resultado_final

    # ------------------------------------------------------------------ #
    #  UTILIDADES                                                          #
    # ------------------------------------------------------------------ #

    def estado_datasets(self) -> Dict:
        """
        Devuelve el estado actual de todos los datasets conocidos.
        Útil para que la interfaz Streamlit sepa qué badge mostrar.
        """
        estado_json = _leer_estado_entrenamiento()
        resultado = {}
        for dataset_id in DATASETS_CONFIG:
            tiene_modelo = dataset_id in self.centinelas
            en_progreso = dataset_id in self._entrenando
            estado_disco = estado_json.get(dataset_id, "sin_modelo")
            
            if tiene_modelo:
                estado = "listo"
            elif en_progreso:
                estado = "entrenando"
            else:
                estado = estado_disco
            
            resultado[dataset_id] = {
                "tiene_modelo": tiene_modelo,
                "estado": estado
            }
        return resultado

    def generar_reporte_md(self, resultado: Dict) -> str:
        if not resultado.get("deteccion"):
            return "Error: No hubo detección."

        tipo = resultado["deteccion"]["tipo_ataque"]
        conf = resultado["deteccion"]["confianza"]
        fallback = resultado.get("fallback_activo", False)
        t_llm = resultado.get("tiempo_normalizacion_llm", 0.0)

        aviso_fallback = ""
        if fallback:
            aviso_fallback = f"""
## ⚠️ MODO FALLBACK LLM ACTIVO
> El sistema no dispone del modelo específico para `{resultado.get('dataset_origen', '?')}`.
> Se ha utilizado el modelo base (Edge-IIoT) con normalización LLM.
> **Precisión puede estar DEGRADADA.** Tiempo extra de normalización: {t_llm:.2f}s
"""

        md = f"""
# REPORTE DE INCIDENTE DE SEGURIDAD - {resultado['id_evento']}
{aviso_fallback}
## 1. Detección (Agente Centinela)
* **Amenaza Detectada:** `{tipo}`
* **Confianza del Modelo:** {conf:.2%}
* **Modelo Usado:** `{resultado['deteccion']['modelo']}`
* **Estado:** {'CRÍTICO' if tipo != 'Normal' else 'SEGURO'}

## 2. Análisis Técnico (Agente Analista)
> {resultado['analisis']['explicacion'] if resultado['analisis'] else 'N/A'}

* **Severidad:** {resultado['analisis']['severidad'] if resultado['analisis'] else 'N/A'}
* **Marco MITRE:** {resultado['analisis']['tactica_mitre'] if resultado['analisis'] else 'N/A'}

## 3. Plan de Mitigación (Agente Respondedor - RAG)
**Acciones Recomendadas (Fuente Oficial):**
{resultado['respuesta']['mitigacion_tecnica'] if resultado['respuesta'] else 'N/A'}

## 4. Auditoría
* **Tiempo de Respuesta Total:** {resultado.get('tiempo_total_procesamiento', 0):.4f}s
* **Tiempo Normalización LLM:** {t_llm:.4f}s
* **Modo:** {'Simulación' if self.modo_simulacion else 'Producción'}
"""
        return md


if __name__ == "__main__":
    import os
    print("=" * 60)
    print("SISTEMA MULTIAGENTE DE CIBERSEGURIDAD IoT")
    print("=" * 60)
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Define la variable de entorno GROQ_API_KEY")
    else:
        sistema = SistemaCiberseguridad(api_key_groq=api_key, modo_simulacion=True)
        print(f"\nEstado de los datasets: {sistema.estado_datasets()}")
