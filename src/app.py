import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="IDS Multiagente IoT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga de Estilos y Utilidades ---
from ui_utils import (
    load_css, badge_html, color_severidad, kpi_card, 
    learning_indicator, centinela_card, analista_card, respondedor_card
)
from logic_utils import limpiar_fuga_datos, format_log_preview, validar_esquema_minimo

load_css("assets/style.css")

# --- Rutas ---
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

from config import DATASETS_CONFIG, DATASET_DEFAULT, AUTODETECCION_ID, DATA_DIR, MODELOS_DIR
from orquestador import SistemaCiberseguridad
from agente_centinela import AgenteCentinela, preparar_datos_desde_csv


# --- Cache ---

@st.cache_resource
def iniciar_sistema():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        # El orquestador ahora se encarga de cargar todos los modelos disponibles
        sistema = SistemaCiberseguridad(
            api_key_groq=api_key,
            modo_simulacion=True
        )
        return sistema
    except Exception as e:
        st.error(f"Error al iniciar el sistema: {e}")
        return None


@st.cache_data
def obtener_todos_los_tipos(dataset_id, col_type):
    """Lee por chunks para extraer únicos sin bloquear la memoria con 1.5M de filas."""
    try:
        ruta_csv = DATASETS_CONFIG[dataset_id]["csv"]
        if not ruta_csv.exists():
            return []
        
        tipos = set()
        # Leemos en trozos de 100k para ser eficientes
        for chunk in pd.read_csv(ruta_csv, usecols=[col_type], chunksize=100_000):
            tipos.update(chunk[col_type].dropna().unique().tolist())
            # Si ya tenemos un número razonable de tipos (ej: 50), paramos pronto
            # para que la sidebar sea instantánea incluso en archivos Gigantes.
            if len(tipos) > 50:
                break
        return sorted(list(tipos))
    except Exception as e:
        logger.error(f"Error obteniendo tipos optimizados de {dataset_id}: {e}")
        return []


@st.cache_data
def cargar_datos(dataset_id):
    """Carga una muestra aleatoria REAL saltando filas en el fichero."""
    try:
        ruta_csv = DATASETS_CONFIG[dataset_id]["csv"]
        if not ruta_csv.exists():
            return None
        
        # Estrategia: Leemos una pequeña muestra aleatoria distribuida por el archivo
        # Para que sea instantáneo, usamos skip_rows con probabilidad
        probabilidad = 0.01 # Leemos el 1% del archivo
        df_sample = pd.read_csv(
            ruta_csv, 
            skiprows=lambda i: i > 0 and random.random() > probabilidad,
            nrows=5000 # Max 5000 registros para la simulación
        )
        return df_sample
    except Exception as e:
        logger.error(f"Error en carga aleatoria de {dataset_id}: {e}")
        # Fallback a lectura simple si el random falla
        return pd.read_csv(ruta_csv, nrows=5000)


def muestra_para_tipo(df, tipo, col_type='Attack_type'):
    if tipo and tipo != "Aleatorio":
        sub = df[df[col_type] == tipo]
        if not sub.empty:
            return sub.sample(1).iloc[0]
    return df.sample(1).iloc[0]



# --- Plantillas de log para inyección manual ---
TEMPLATES_PATH = DATA_DIR / "templates_ataques.json"
try:
    import json
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        _templates_raw = json.load(f)
        # Convertir a string formateado para el textarea
        LOG_TEMPLATES = {k: json.dumps(v, indent=2, ensure_ascii=False) for k, v in _templates_raw.items()}
except Exception as e:
    logger.error(f"Error cargando plantillas de ataque: {e}")
    LOG_TEMPLATES = {"⚠️ Error cargando plantillas": "{}"}

# --- Estado de sesión ---
if "historial" not in st.session_state:
    st.session_state.historial = []
if "simulando" not in st.session_state:
    st.session_state.simulando = False
if "tipo_seleccionado" not in st.session_state:
    st.session_state.tipo_seleccionado = "Aleatorio"
if "log_manual" not in st.session_state:
    st.session_state.log_manual = ""
if "modo_manual" not in st.session_state:
    st.session_state.modo_manual = False
if "trace_log" not in st.session_state:
    st.session_state.trace_log = []


# ============================================================
# FUNCIONES DE ADAPTACIÓN (HOT-SWAP)
# ============================================================

def procesar_nuevo_dataset(archivo_subido, sistema):
    """Maneja la carga, inferencia de esquema y entrenamiento de un nuevo dataset experto."""
    nombre_ds = archivo_subido.name.split('.')[0]
    ruta_guardado = DATA_DIR / archivo_subido.name
    
    with st.status(f"⚙️ Adaptando sistema a '{nombre_ds}'...", expanded=True) as status_box:
        st.write(f"💾 Guardando {archivo_subido.name}...")
        with open(ruta_guardado, "wb") as f:
            f.write(archivo_subido.getbuffer())
        
        # 1. Inferir esquema semántico
        st.write("🧠 Infiriendo esquema del dataset...")
        df_temp = pd.read_csv(ruta_guardado, nrows=50)
        esquema = sistema.normalizador.inferir_esquema_dataset(df_temp)
        
        # 2. Registrar en la configuración global (en caliente)
        DATASETS_CONFIG[nombre_ds] = {
            "csv": ruta_guardado,
            "modelo": MODELOS_DIR / f"xgboost_{nombre_ds}.pkl",
            "col_label": esquema["col_label"],
            "col_type": esquema["col_type"]
        }
        
        # 3. Entrenar Modelo Experto
        st.write(f"🔧 Entrenando Modelo Experto XGBoost...")
        
        # Cargamos datos para entrenamiento (muestra representativa)
        df_full = pd.read_csv(ruta_guardado, low_memory=False)
        n_muestras = min(10000, len(df_full))
        df_train = df_full.sample(n=n_muestras, random_state=42)
        
        X = df_train.drop(columns=[esquema["col_label"], esquema["col_type"]], errors='ignore')
        # Limpieza básica de columnas no numéricas para el experto
        X = X.select_dtypes(include=[np.number])
        y = df_train[esquema["col_type"]]
        
        nuevo_centinela = AgenteCentinela()
        nuevo_centinela.entrenar(X, y)
        
        nuevo_centinela.guardar_modelo(str(DATASETS_CONFIG[nombre_ds]["modelo"]))
        
        # 4. Notificar al Orquestador
        sistema.registrar_nuevo_centinela(nombre_ds, nuevo_centinela)
        
        status_box.update(label=f"✅ ¡Sistema Adaptado! Contexto '{nombre_ds}' listo.", state="complete")
        time.sleep(1.5)
    
    # Cambiar el dataset actual al nuevo automáticamente
    st.session_state.dataset_actual = nombre_ds
    st.rerun()



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ Panel de Control")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    sistema = iniciar_sistema()

    # --- CARGA DE NUEVOS DATASETS (HOT-SWAP) ---
    st.markdown("<p class='section-title'>Adaptación Autónoma</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Subir nuevo contexto (CSV)", 
        type=["csv"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if st.button("Adaptar Sistema al Nuevo Contexto", use_container_width=True):
            procesar_nuevo_dataset(uploaded_file, sistema)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


    # --- Selector de Contexto de Red ---
    st.markdown("<p class='section-title'>Contexto de Red Detectado</p>", unsafe_allow_html=True)
    dataset_ids = list(DATASETS_CONFIG.keys())
    # Guardamos la selección en el estado para que no se pierda
    if "dataset_actual" not in st.session_state:
        # Por defecto, Autodetección: el sistema no asume ningún contexto de red
        st.session_state.dataset_actual = AUTODETECCION_ID

    idx_default = dataset_ids.index(st.session_state.dataset_actual) if st.session_state.dataset_actual in dataset_ids else 0
    dataset_sel = st.selectbox(
        "Origen del tráfico",
        dataset_ids,
        index=idx_default,
        help="'Autodetección' activa el LLM Normalizador para analizar cualquier log sin configuración previa. Los contextos conocidos usan el modelo experto específico."
    )
    st.session_state.dataset_actual = dataset_sel

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Estado de agentes y Monitor de Background
    st.markdown("<p class='section-title'>Estado de Modelos</p>", unsafe_allow_html=True)
    if sistema:
        estados = sistema.estado_datasets()
        
        # Mostramos una pequeña tabla de estados
        for d_id, info in estados.items():
            est = info["estado"]
            if est == "listo":
                color, label = "ok", "✓ Listo"
            elif est == "entrenando":
                color, label = "warning", "⚙️ Entrenando..."
            elif est == "sin_dataset":
                color, label = "critical", "❗ CSV no hallado"
            else:
                color, label = "critical", "⚠ Fallback LLM"
            
            # Resaltar el dataset actual
            prefijo = "**" if d_id == dataset_sel else ""
            st.markdown(f"{prefijo}{d_id}: {badge_html(label, color)}{prefijo}", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Métricas del modelo seleccionado (si está listo)
        centinela_sel = sistema.centinelas.get(dataset_sel)
        if centinela_sel and centinela_sel.esta_entrenado:
            with st.expander(f"Métricas: {dataset_sel}"):
                m = centinela_sel.metricas
                st.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
                st.metric("F1 (weighted)", f"{m.get('f1_weighted', 0):.4f}")
    else:
        st.markdown(badge_html("✗ Sistema offline", "critical"), unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Indicador de aprendizaje (Fase 3 en background) si estamos en autodetección
    if st.session_state.dataset_actual == AUTODETECCION_ID:
        learning_indicator()

    # --- NUEVO: Selector de Modo (Manual vs Automático) ---
    st.markdown("<p class='section-title'>Modo de Simulación</p>", unsafe_allow_html=True)
    modo = st.radio(
        "Método de entrada", 
        ["Simulador Automático", "Inyección Manual (JSON)"],
        label_visibility="collapsed"
    )
    st.session_state.modo_manual = (modo == "Inyección Manual (JSON)")

    if not st.session_state.modo_manual:
        # MODO AUTOMÁTICO (CSV)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Simulador de tráfico</p>", unsafe_allow_html=True)

        # En modo Autodetección no hay CSV de referencia: guiar al usuario al modo manual
        if dataset_sel == AUTODETECCION_ID:
            st.markdown("""
            <div style='background:#1a2233; border:1px solid #388bfd55; border-radius:8px;
                        padding:12px 14px; font-size:0.82rem; color:#79c0ff; line-height:1.6;'>
            🤖 <b>Modo Autodetección activo</b><br>
            El sistema no asume ningún origen de red conocido.<br>
            Usa <b>Inyección Manual (JSON)</b> para pegar tu log y el LLM normalizará
            automáticamente cualquier formato.
            </div>
            """, unsafe_allow_html=True)
            df = None
        else:
            df = cargar_datos(dataset_sel)

        if df is not None:
            # Petición a la 'Memoria Semántica' del orquestador
            esquema = sistema.obtener_columnas_esquema(dataset_sel)
            col_type = esquema["col_type"]
            col_label = esquema["col_label"]

            lista_ataques = obtener_todos_los_tipos(dataset_sel, col_type)
            tipos = ["Aleatorio"] + lista_ataques
            tipo_sel = st.selectbox("Tipo de evento", tipos, key="tipo_sel")

            if st.button("🔍 Simular evento", type="primary", use_container_width=True):
                st.session_state.simulando = True
                st.session_state.tipo_seleccionado = tipo_sel
                st.rerun()
        else:
            st.warning("⚠️ CSV no disponible para este contexto. Usa el modo 'Inyección Manual (JSON)'.")
    else:
        # MODO MANUAL (JSON)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Terminal de Inyección Manual</p>", unsafe_allow_html=True)

        # --- Botones de Plantilla ---
        st.markdown("<span style='font-size:0.78rem; color:#8b949e;'>Cargar plantilla de ejemplo:</span>", unsafe_allow_html=True)
        cols_tpl = st.columns(2)
        template_names = list(LOG_TEMPLATES.keys())
        for i, tpl_name in enumerate(template_names):
            with cols_tpl[i % 2]:
                if st.button(tpl_name, key=f"tpl_{i}", use_container_width=True):
                    st.session_state.log_manual = LOG_TEMPLATES[tpl_name]
                    st.rerun()

        log_input = st.text_area(
            "Pegar log JSON aquí:", 
            value=st.session_state.log_manual, 
            height=150,
            placeholder='{"frame.len": 60, "tcp.flags": 2, ...}'
        )
        st.session_state.log_manual = log_input

        if st.button("🚀 Inyectar Tráfico Real", type="primary", use_container_width=True):
            if not log_input.strip():
                st.error("Por favor, pega un log válido.")
            else:
                # Validación de esquema mínimo
                try:
                    features_dict_manual = json.loads(log_input)
                    if not validar_esquema_minimo(features_dict_manual):
                        st.warning("⚠️ El log JSON parece incompleto. Asegúrate de incluir suficientes campos técnicos.")
                    
                    st.session_state.simulando = True
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("❌ El formato JSON no es válido. Revisa las comas y comillas.")

    if not sistema:
        st.stop()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Historial rápido en sidebar
    if st.session_state.historial:
        st.markdown("<p class='section-title'>Últimas detecciones</p>", unsafe_allow_html=True)
        for h in st.session_state.historial[-5:][::-1]:
            ico = "🔴" if h["tipo"] != "Normal" else "🟢"
            st.markdown(
                f"{ico} `{h['tipo']}` — {h['confianza']:.0%}  \n"
                f"<span style='font-size:0.72rem;color:#8b949e'>{h['hora']}</span>",
                unsafe_allow_html=True
            )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<h1 style='font-size:1.9rem; font-weight:700; margin-bottom:2px;'>
    Sistema de Detección de Intrusiones IoT
</h1>
<p style='color:#8b949e; font-size:0.9rem; margin-top:0;'>
    Pipeline multiagente: XGBoost · LLM (Groq) · RAG (MITRE CAPEC)
</p>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ============================================================
# KPIs de sesión
# ============================================================

total = len(st.session_state.historial)
amenazas = sum(1 for h in st.session_state.historial if h["tipo"] != "Normal")
porc_amenaza = (amenazas / total * 100) if total > 0 else 0
tiempo_prom = (
    sum(h.get("tiempo", 0) for h in st.session_state.historial) / total
    if total > 0 else 0
)
tipo_frecuente = "—"
if amenazas > 0:
    detecciones = [h["tipo"] for h in st.session_state.historial if h["tipo"] != "Normal"]
    tipo_frecuente = Counter(detecciones).most_common(1)[0][0]

col1, col2, col3, col4 = st.columns(4)
kpis = [
    (col1, str(total), "Eventos analizados"),
    (col2, f"{amenazas}", "Amenazas detectadas"),
    (col3, f"{porc_amenaza:.1f}%", "Tasa de amenazas"),
    (col4, f"{tiempo_prom:.2f}s" if total > 0 else "—", "Tiempo promedio respuesta"),
]

for col, val, lbl in kpis:
    with col:
        kpi_card(val, lbl)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SIMULACIÓN
# ============================================================

if st.session_state.simulando:
    
    if st.session_state.modo_manual:
        # PROCESAR ENTRADA MANUAL
        try:
            features_dict = json.loads(st.session_state.log_manual)
            tipo_original = "Manual/Desconocido"
        except Exception as e:
            st.error(f"Error en formato JSON: {e}")
            st.session_state.simulando = False
            st.stop()
    else:
        # PROCESAR DESDE DATASET (CSV)
        esquema = sistema.obtener_columnas_esquema(st.session_state.dataset_actual)
        col_type = esquema["col_type"]
        col_label = esquema["col_label"]

        muestra = muestra_para_tipo(df, st.session_state.tipo_seleccionado, col_type=col_type)
        
        # ELIMINACIÓN DE VARIABLES FILTRADAS (Data Leakage Mitigation) usando logic_utils
        features_dict = limpiar_fuga_datos(muestra, col_label, col_type)
        tipo_original = muestra[col_type] if col_type in muestra else "N/A"

    # Construimos el string del log limitado para la vista previa usando logic_utils
    log_preview = format_log_preview(features_dict)
    
    # Caja verde con scroll vertical y wrap, agrupada con su título para evitar márgenes de Streamlit
    st.markdown(f"""
<p class='section-title' style='margin-bottom: 6px;'>Tráfico de red interceptado</p>
<div style="background-color: #0d1117; padding: 5px 15px 15px 15px; margin-bottom: 15px; border-radius: 10px; border: 1px solid #30363d; font-family: monospace; font-size: 0.85rem; color: #7ee787; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; line-height: 1.4;">
{log_preview}
</div>
    """, unsafe_allow_html=True)
    
    with st.expander("Ver todos los metadatos técnicos (Full JSON)"):
        log_str = json.dumps(features_dict, indent=2)
        st.code(log_str, language="json")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Barra de progreso
    progress = st.progress(0)
    status = st.empty()

    try:
        status.markdown("🔎 &nbsp; Centinela inspeccionando paquetes...")
        progress.progress(25)
        time.sleep(0.4)

        t_inicio = time.time()
        # Enviamos el log y el ID del dataset actual
        resultado = sistema.procesar_trafico(features_dict, dataset_id=st.session_state.dataset_actual)
        t_total = time.time() - t_inicio

        status.markdown("🧠 &nbsp; Analista LLM generando informe técnico...")
        progress.progress(65)
        time.sleep(0.35)

        status.markdown("📖 &nbsp; Respondedor consultando base MITRE CAPEC...")
        progress.progress(90)
        time.sleep(0.35)

        progress.progress(100)
        time.sleep(0.15)
    except Exception as e:
        st.error(f"Error crítico en el pipeline: {e}")
        resultado = {}
        t_total = 0
    finally:
        progress.empty()
        status.empty()
        st.session_state.simulando = False

    deteccion = resultado.get("deteccion") or {}
    analisis = resultado.get("analisis") or {}
    respuesta = resultado.get("respuesta") or {}

    tipo_ataque = deteccion.get("tipo_ataque", "Error en detección")
    confianza = deteccion.get("confianza", 0.0)
    hora_actual = datetime.now().strftime("%H:%M:%S")
    es_normal = (tipo_ataque.lower() == "normal" or tipo_ataque.lower() == "0")

    # Refinamiento semántico: Si el analista dedujo un tipo más específico en modo fallback, lo usamos para el historial
    if resultado.get("fallback_activo") and analisis.get("tipo_ataque") and not es_normal:
        tipo_ataque = analisis.get("tipo_ataque")

    # --- Construir Trace de Ejecución Semántica ---
    _trace = []
    _ds_id = st.session_state.dataset_actual
    _fallback = resultado.get("fallback_activo", False)
    _t_norm = resultado.get("tiempo_normalizacion_llm", 0)
    _modelo_usado = resultado.get("modelo_usado", "desconocido")

    _trace.append(("INFO", "Orquestador", f"Log recibido. Contexto de red: '{_ds_id}'."))
    if _fallback:
        _trace.append(("WARN", "Orquestador", "Esquema desconocido o sin modelo experto disponible -> activando pipeline de Supervivencia."))
        _trace.append(("EXEC", "Agente Normalizador", f"Traduciendo {len(features_dict)} campos al esquema canónico universal... completado en {_t_norm:.2f}s."))
        _trace.append(("EXEC", "Agente Centinela", f"Clasificando con modelo universal ({_modelo_usado}): '{tipo_ataque}' — confianza {confianza:.1%}."))
    else:
        _trace.append(("INFO", "Orquestador", f"Modelo experto disponible para '{_ds_id}' -> enrutando a pipeline nativo."))
        _trace.append(("EXEC", "Agente Centinela", f"Clasificación directa: '{tipo_ataque}' — confianza {confianza:.1%}."))

    if analisis:
        _trace.append(("EXEC", "Agente Analista", f"Análisis MITRE completado. Táctica: {analisis.get('tactica_mitre', '—')} | Severidad: {analisis.get('severidad', '—')}."))
    if respuesta:
        n_fuentes = len(resultado.get('respuesta', {}).get('fuentes_cap', []))
        _trace.append(("EXEC", "Agente Respondedor", f"Mitigaciones generadas consultando {n_fuentes} fuente(s) CAPEC."))
    _trace.append(("DONE", "Orquestador", f"Pipeline completado en {t_total:.2f}s. Resultado almacenado en historial."))

    st.session_state.trace_log = _trace

    # Guardar en historial con seguridad
    st.session_state.historial.append({
        "hora": hora_actual,
        "tipo": tipo_ataque,
        "confianza": confianza if confianza is not None else 0.0,
        "severidad": analisis.get("severidad", "—") if analisis else "—",
        "mitre": analisis.get("tactica_mitre", "—") if analisis else "—",
        "tiempo": t_total,
        "fallback": resultado.get("fallback_activo", False)
    })

    # --- PANELES DE RESULTADO ---
    st.markdown("<p class='section-title' style='margin-bottom:12px;'>Resultado del análisis</p>",
                unsafe_allow_html=True)

    cola, colb, colc = st.columns(3)

    # 👁 Panel Centinela
    with cola:
        centinela_card(
            tipo_ataque, confianza, es_normal, 
            resultado.get('dataset_origen', 'n/a'), 
            resultado.get("fallback_activo")
        )

    #  Panel Analista
    with colb:
        analista_card(analisis, es_normal, t_total)

    # 🛡 Panel Respondedor
    with colc:
        respondedor_card(
            respuesta, 
            resultado.get('fallback_activo'), 
            resultado.get('tiempo_normalizacion_llm', 0)
        )
        
        if respuesta and resultado.get('respuesta', {}).get('fuentes_cap'):
            with st.expander("Ver fuentes RAG consultadas"):
                for f in list(dict.fromkeys(resultado['respuesta']['fuentes_cap'])):
                    st.markdown(f"- `{f}`")

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TRACE DE EJECUCIÓN SEMÁNTICA
# ============================================================

if st.session_state.get("trace_log"):
    with st.expander("Trace de Ejecución Semántica", expanded=False):
        st.markdown(
            "<p style='color:#8b949e; font-size:0.82rem; margin-bottom:12px;'>"
            "Registro del sistema: Secuencia de decisiones multiagente."
            "</p>",
            unsafe_allow_html=True
        )
        _color_map = {
            "INFO": "#58a6ff",  # Azul
            "WARN": "#f0883e",  # Naranja
            "EXEC": "#d29922",  # Amarillo
            "DONE": "#3fb950",  # Verde
        }
        trace_html = "<div style='font-family: monospace; font-size: 0.83rem; line-height: 1.9; background-color: #0d1117; padding: 15px; border-radius: 6px; border: 1px solid #30363d;'>"
        for i, (tag, agente, mensaje) in enumerate(st.session_state.trace_log):
            color = _color_map.get(tag, "#c9d1d9")
            trace_html += (
                f"<div style='display:flex; align-items:baseline; gap:12px;'>"
                f"<span style='color:{color}; font-weight:bold; min-width:45px;'>[{tag}]</span>"
                f"<span style='color:#c9d1d9; font-weight:600; min-width:160px;'>{agente}</span>"
                f"<span style='color:#8b949e;'>{mensaje}</span>"
                f"</div>"
            )
        trace_html += "</div>"
        st.markdown(trace_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# VISUALIZACIONES DE SESIÓN + CICLO DE VIDA DEL SISTEMA
# ============================================================

if len(st.session_state.historial) >= 2:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title' style='margin-bottom:14px;'>Actividad de la sesión</p>",
                unsafe_allow_html=True)

    viz1, viz2 = st.columns([1, 1])

    # Gráfico distribución de ataques
    with viz1:
        hist_df = pd.DataFrame(st.session_state.historial)
        counts = hist_df["tipo"].value_counts().reset_index()
        counts.columns = ["Tipo", "Eventos"]

        colores = {
            "Normal": "#3fb950",
            "DDoS_UDP": "#f85149", "DDoS_TCP": "#f85149", "DDoS_HTTP": "#f85149", "DDoS_ICMP": "#f85149",
            "SQL_injection": "#f0883e", "Backdoor": "#f0883e", "Ransomware": "#f0883e",
            "Port_Scanning": "#d29922", "Vulnerability_scanner": "#d29922", "Fingerprinting": "#d29922",
            "MITM": "#a371f7", "XSS": "#a371f7", "Uploading": "#79c0ff", "Password": "#79c0ff",
        }
        bar_colors = [colores.get(t, "#58a6ff") for t in counts["Tipo"]]

        fig_bar = go.Figure(go.Bar(
            x=counts["Tipo"],
            y=counts["Eventos"],
            marker_color=bar_colors,
            text=counts["Eventos"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Distribución de eventos detectados",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", size=12),
            xaxis=dict(tickangle=-30, gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=20, r=20, t=50, b=60),
            title_font=dict(size=13, color="#e6edf3"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Gauge de confianza del último evento
    with viz2:
        ultimo = st.session_state.historial[-1]
        conf_val = ultimo.get("confianza", 0.0)
        conf_pct = conf_val * 100
        tipo_ult = ultimo.get("tipo", "Normal")
        col_aguja = "#f85149" if tipo_ult != "Normal" else "#3fb950"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=conf_pct,
            number={"suffix": "%", "font": {"size": 38, "color": col_aguja}},
            delta={"reference": 80, "increasing": {"color": col_aguja}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": col_aguja},
                "bgcolor": "#0d1117",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 60], "color": "#21262d"},
                    {"range": [60, 85], "color": "#1e2d2b"},
                    {"range": [85, 100], "color": "#1a1f29"},
                ],
                "threshold": {
                    "line": {"color": "#58a6ff", "width": 2},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
            title={"text": f"Confianza — último evento<br><span style='font-size:0.8em;color:#8b949e'>{tipo_ult}</span>",
                   "font": {"size": 13, "color": "#e6edf3"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            margin=dict(l=20, r=20, t=60, b=20),
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ============================================================
# CICLO DE VIDA DEL SISTEMA — Línea temporal de latencia por fase
# ============================================================

if len(st.session_state.historial) >= 1:
    hist_df_full = pd.DataFrame(st.session_state.historial)
    tiene_fases = "fallback" in hist_df_full.columns and hist_df_full["fallback"].any()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-title' style='margin-bottom:14px;'>📡 Ciclo de Vida del Sistema — Pipeline de Alta Disponibilidad</p>",
        unsafe_allow_html=True
    )

    # ── Gráfico temporal de latencia coloreado por fase ────────────────────
    hist_df_full["evento_num"] = range(1, len(hist_df_full) + 1)
    hist_df_full["fase"] = hist_df_full["fallback"].apply(
        lambda x: "⚠️ Fase 1 — Modo Supervivencia (LLM Fallback)" if x
        else "✅ Fase 3 — Modo Experto (XGBoost Nativo)"
    )
    hist_df_full["latencia_ms"] = hist_df_full["tiempo"].apply(lambda x: round(x * 1000, 1))

    col_fase = {
        "⚠️ Fase 1 — Modo Supervivencia (LLM Fallback)": "#f0883e",
        "✅ Fase 3 — Modo Experto (XGBoost Nativo)": "#3fb950",
    }

    fig_lifecycle = go.Figure()
    for fase_name, color in col_fase.items():
        sub = hist_df_full[hist_df_full["fase"] == fase_name]
        if sub.empty:
            continue
        fig_lifecycle.add_trace(go.Scatter(
            x=sub["evento_num"],
            y=sub["latencia_ms"],
            mode="markers+lines",
            name=fase_name,
            line=dict(color=color, width=2, dash="solid"),
            marker=dict(color=color, size=9, symbol="circle",
                        line=dict(color="#0d1117", width=1.5)),
            hovertemplate=(
                "<b>Evento #%{x}</b><br>"
                "Latencia: %{y:.1f} ms<br>"
                f"Fase: {fase_name}"
                "<extra></extra>"
            ),
        ))

    # Línea de referencia: latencia objetivo (sub-ms clasificación pura)
    fig_lifecycle.add_hline(
        y=100, line_dash="dot", line_color="#58a6ff", line_width=1.5,
        annotation_text="100 ms — umbral de alerta",
        annotation_position="top right",
        annotation_font_color="#58a6ff",
    )

    fig_lifecycle.update_layout(
        title=dict(
            text="Latencia de respuesta por evento (ms) — transición Fase 1→3",
            font=dict(size=13, color="#e6edf3")
        ),
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        xaxis=dict(
            title="Nº de evento",
            gridcolor="#21262d",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            title="Latencia total (ms)",
            gridcolor="#21262d",
            type="log",  # log para ver bien la diferencia de órdenes de magnitud
        ),
        legend=dict(
            bgcolor="#0d1117",
            bordercolor="#30363d",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=50, r=20, t=80, b=50),
        height=320,
    )
    st.plotly_chart(fig_lifecycle, use_container_width=True)

    if not tiene_fases:
        st.info(
            "💡Selecciona un dataset sin modelo (ej. TON_IoT antes de entrenarlo) "
            "para ver la transición Fase 1 → Fase 3 en este gráfico."
        )




# ============================================================
# HISTORIAL DE SIMULACIONES
# ============================================================

if st.session_state.historial:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title' style='margin-bottom:12px;'>Historial de simulaciones</p>",
                unsafe_allow_html=True)

    df_hist_raw = pd.DataFrame(st.session_state.historial)

    # Columnas base garantizadas
    cols_mostrar = ["hora", "tipo", "confianza", "severidad", "mitre", "tiempo", "fallback"]
    cols_disponibles = [c for c in cols_mostrar if c in df_hist_raw.columns]
    df_hist = df_hist_raw[cols_disponibles].copy()

    renombrar = {
        "hora": "Hora",
        "tipo": "Tipo de ataque",
        "confianza": "Confianza",
        "severidad": "Severidad",
        "mitre": "MITRE",
        "tiempo": "t (s)",
        "fallback": "Modo",
    }
    df_hist.rename(columns=renombrar, inplace=True)
    if "Confianza" in df_hist.columns:
        df_hist["Confianza"] = df_hist["Confianza"].apply(lambda x: f"{x:.1%}")
    if "t (s)" in df_hist.columns:
        df_hist["t (s)"] = df_hist["t (s)"].apply(
            lambda x: f"{x:.2f}s" if isinstance(x, float) else x
        )
    if "Modo" in df_hist.columns:
        df_hist["Modo"] = df_hist["Modo"].apply(
            lambda x: "⚠️ Supervivencia (LLM)" if x else "✅ Experto (Nativo)"
        )

    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
    col_dl, col_clear = st.columns([3, 1])
    with col_dl:
        st.download_button(
            label="⬇ Descargar historial (.csv)",
            data=csv_bytes,
            file_name=f"ids_sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with col_clear:
        if st.button("🗑 Limpiar historial"):
            st.session_state.historial = []
            st.rerun()


# ============================================================
# RESULTADOS  — Trade-off Alta Disponibilidad (TFM)
# ============================================================

from config import RESULTADOS_DIR

RUTA_CSV_COMP  = RESULTADOS_DIR / "comparativa_fase1_vs_fase3.csv"
RUTA_JSON_COMP = RESULTADOS_DIR / "comparativa_llm_vs_xgboost.json"

if RUTA_CSV_COMP.exists() or RUTA_JSON_COMP.exists():
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-title' style='margin-bottom:16px;'> Resultados Empíricos — Trade-off de Alta Disponibilidad</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#8b949e; font-size:0.85rem; margin-bottom:16px;'>"
        "Comparativa empírica entre el <b style='color:#f0883e'>Modo Supervivencia (Fase 1)</b> — "
        "Fallback LLM con detección binaria — y el "
        "<b style='color:#3fb950'>Modo Experto (Fase 3)</b> — modelo XGBoost nativo. "
        "Estos datos constituyen la validación del trade-off arquitectónico del TFM."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Cargar el JSON si existe (resultado más reciente) ────────────────
    datos_json = None
    if RUTA_JSON_COMP.exists():
        try:
            with open(RUTA_JSON_COMP, "r", encoding="utf-8") as _f:
                datos_json = json.load(_f)
        except Exception:
            pass

    if datos_json and "fase3_nativo" in datos_json and "fase1_fallback_llm" in datos_json:
        fase3 = datos_json["fase3_nativo"]
        fase1 = datos_json["fase1_fallback_llm"]
        tof   = datos_json.get("trade_off", {})
        ds_id = datos_json.get("dataset_id", "")

        # ── KPI cards del trade-off ───────────────────────────────────────
        kc1, kc2, kc3, kc4 = st.columns(4)

        kpis_comp = [
            (kc1,
             f"{fase3.get('accuracy', fase3.get('f1_macro', 0)):.1%}",
             "F1/Acc Nativo",
             "#3fb950"),
            (kc2,
             f"{fase1.get('recall', fase1.get('f1_binario', 0)):.1%}",
             "Recall Detección Supervivencia",
             "#f0883e"),
            (kc3,
             f"+{tof.get('sobrecoste_latencia_media_ms', 0):.0f} ms",
             "Sobrecoste Latencia Media",
             "#d29922"),
            (kc4,
             f"{fase3.get('latencia', {}).get('media_ms', fase3.get('latencia_media_ms', 0)):.2f} ms",
             "Latencia Nativo",
             "#58a6ff"),
        ]
        for col_k, val_k, lbl_k, clr_k in kpis_comp:
            with col_k:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color:{clr_k}">{val_k}</div>
                    <div class="kpi-label">{lbl_k}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gráfico comparativo de métricas ──────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            # Comparativa de rendimiento (F1 / accuracy)
            labels_met = ["Fase 3 — Nativo<br>(multiclase)", "Fase 1 — Supervivencia<br>(binaria)"]
            val_f1     = [
                fase3.get("f1_macro", fase3.get("accuracy", 0)),
                fase1.get("f1_binario", fase1.get("f1_macro", 0)),
            ]
            colores_comp = ["#3fb950", "#f0883e"]

            fig_met = go.Figure(go.Bar(
                x=labels_met,
                y=val_f1,
                marker_color=colores_comp,
                text=[f"{v:.1%}" for v in val_f1],
                textposition="inside",
                textfont=dict(size=14, color="white"),
            ))
            fig_met.update_layout(
                title=dict(text=f"F1-Score — {ds_id}", font=dict(size=12, color="#e6edf3")),
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", size=11),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", range=[0, 1.1], tickformat=".0%"),
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
                height=280,
            )
            st.plotly_chart(fig_met, use_container_width=True)

        with ch2:
            # Comparativa de latencia (log scale)
            lat3 = fase3.get("latencia", {})
            lat1 = fase1.get("latencia", {})

            # Compatibilidad con formato antiguo y nuevo
            lat3_med = lat3.get("media_ms", fase3.get("latencia_media_ms", 0))
            lat1_med = lat1.get("media_ms", fase1.get("latencia_media_ms", 0))
            lat3_p95 = lat3.get("p95_ms", lat3_med)
            lat1_p95 = lat1.get("p95_ms", lat1_med)

            categorias_lat = ["Latencia media", "Latencia p95"]
            vals_fase3 = [lat3_med, lat3_p95]
            vals_fase1 = [lat1_med, lat1_p95]

            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                name="Fase 3 — Nativo",
                x=categorias_lat,
                y=vals_fase3,
                marker_color="#3fb950",
                text=[f"{v:.1f} ms" for v in vals_fase3],
                textposition="inside",
                textfont=dict(color="white"),
            ))
            fig_lat.add_trace(go.Bar(
                name="Fase 1 — Supervivencia",
                x=categorias_lat,
                y=vals_fase1,
                marker_color="#f0883e",
                text=[f"{v:.1f} ms" for v in vals_fase1],
                textposition="inside",
                textfont=dict(color="white"),
            ))
            fig_lat.update_layout(
                title=dict(text="Latencia (ms) — escala logarítmica", font=dict(size=12, color="#e6edf3")),
                barmode="group",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", size=11),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(
                    gridcolor="#21262d",
                    type="log",
                    title="ms (log)",
                ),
                legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
                margin=dict(l=20, r=20, t=50, b=20),
                height=280,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # ── Conclusión automática ─────────────────────────────────────────
        if tof.get("conclusion"):
            color_concl = "#f0883e" if "LIMITADA" in tof["conclusion"] else (
                "#d29922" if "MODERADA" in tof["conclusion"] else "#3fb950"
            )
            st.markdown(
                f"<div style='background:#161b22; border-left:4px solid {color_concl}; "
                f"padding:14px 18px; border-radius:0 8px 8px 0; margin-top:10px;'>"
                f"<p style='color:{color_concl}; font-size:0.78rem; font-weight:700; "
                f"text-transform:uppercase; letter-spacing:.05em; margin:0 0 6px;'>Veredicto Arquitectónico</p>"
                f"<p style='color:#c9d1d9; font-size:0.84rem; margin:0; line-height:1.6;'>{tof['conclusion']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Tabla CSV completa (expandible) ──────────────────────────────────
    if RUTA_CSV_COMP.exists():
        with st.expander("🗂 Ver tabla completa de registros de evaluación"):
            # Evitamos crashes si el CSV se escribió de forma asíncrona corrupta
            df_comp = pd.read_csv(RUTA_CSV_COMP, on_bad_lines='skip')
            st.dataframe(df_comp, hide_index=True)
            with open(RUTA_CSV_COMP, "rb") as _file:
                st.download_button(
                    label="⬇ Descargar resultados completos (.csv)",
                    data=_file,
                    file_name="trade_off_alta_disponibilidad.csv",
                    mime="text/csv",
                )



