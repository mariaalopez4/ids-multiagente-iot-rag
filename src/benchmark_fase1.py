"""
benchmark_fase1.py
====================================
Experimento 2: Survival Mode (Phase 1)

Parte 1: Clasificacion Binaria del modelo Fallback XGBoost
  - Dataset combinado de los 3 benchmarks (mismo mapa que generar_features_comunes.py)
  - Metricas: Accuracy, F1-Weighted (binario: Ataque / Normal)
  - Latencia de inferencia en microsegundos

Parte 2: Clasificacion Multiclase del LLM Analista (prompt constreñido)
  - N_LLM_SAMPLES muestras de ataque por dataset (solo ataques, no Normal)
  - Prompt constreñido a TAXONOMIA_LLM (6 categorias semanticas)
  - Ground truth mapeado a la misma taxonomia para comparacion justa
  - Metricas: F1-Macro, F1-Weighted, Accuracy

Ejecutar desde la raiz del proyecto:
  python src/benchmark_fase1.py
"""

import sys
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from groq import Groq
from agente_centinela import AgenteCentinela
from config import DATASETS_CONFIG, FEATURES_COMUNES, MODELO_FALLBACK, RESULTADOS_DIR, MODELO_ANALISTA
from generar_features_comunes import (
    MAPA_EQUIVALENCIAS, LABEL_COLS,
    extraer_features_dataset, normalizar_clase
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [FASE1] - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark_fase1")

# ─── Parametros ───────────────────────────────────────────────────────────────
N_BIN_SAMPLES = 10_000   # muestras para F1 binario
N_LLM_SAMPLES = 15       # llamadas LLM por dataset (solo ataques)
RANDOM_STATE  = 42

# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMIA SEMANTICA para la evaluacion LLM (6 categorias cruzando los 3 datasets)
# Permite comparar ground truth y predicciones LLM en un vocabulario comun.
# ─────────────────────────────────────────────────────────────────────────────
TAXONOMIA_LLM = {
    # DDoS / DoS  (Edge-IIoT, TON_IoT, Bot-IoT)
    "ddos": "DDoS/DoS",       "ddos_http": "DDoS/DoS",  "ddos_icmp": "DDoS/DoS",
    "ddos_tcp": "DDoS/DoS",   "ddos_udp": "DDoS/DoS",   "dos": "DDoS/DoS",
    # Bot-IoT usa exactamente estas etiquetas:
    "http": "DDoS/DoS",       "tcp": "DDoS/DoS",        "udp": "DDoS/DoS",
    "syn": "DDoS/DoS",
    # Scanning / Reconnaissance
    "scanning": "Scanning",        "port_scanning": "Scanning",
    "fingerprinting": "Scanning",  "reconnaissance": "Scanning",
    "vulnerability_scanner": "Scanning",
    # Injection (SQL, XSS)
    "injection": "Injection",    "sql_injection": "Injection", "xss": "Injection",
    # Malware / Intrusion / Theft
    "backdoor": "Malware",  "ransomware": "Malware",  "uploading": "Malware",
    "theft": "Malware",
    # Credential / MITM
    "password": "Credential/MITM",  "mitm": "Credential/MITM",
    # Normal  (todos los datasets)
    "normal": "Normal",  "0": "Normal",  "benign": "Normal",  "legitimate": "Normal",
}

CATEGORIAS_LLM = ["DDoS/DoS", "Scanning", "Injection", "Malware", "Credential/MITM", "Normal"]

# Descripciones semánticas de las features canónicas para enriquecer el contexto del LLM
FEATURE_DESCRIPTIONS = {
    "bytes_src":        "Source bytes sent (high → DDoS/DoS, low → scanning/injection)",
    "bytes_dst":        "Destination bytes received",
    "pkts_src":         "Source packet count (high → flood attack)",
    "pkts_dst":         "Destination packet count",
    "avg_packet_size":  "Average packet size in bytes",
    "puerto_src":       "Source port (0 or unusual → scanning; 443/80 → web attack)",
    "puerto_dst":       "Destination port (22=SSH, 80=HTTP, 3306=MySQL, 23=Telnet)",
    "duracion":         "Flow duration in seconds (very short → scan; very long → exfiltration)",
    "flag_syn":         "TCP SYN flag count (high → SYN flood / port scan)",
    "flag_fin":         "TCP FIN flag (connection teardown)",
    "flag_rst":         "TCP RST flag (abrupt close, firewall reject or scan response)",
    "dns_tipo":         "DNS query type (255=ANY → DNS amplification; 28=AAAA → recon)",
    "http_codigo":      "HTTP status code (200=OK, 404=scanning, 500=injection attempt)",
    "http_bytes_resp":  "HTTP response body size (large → data exfiltration)",
    "icmp_seq":         "ICMP sequence number (non-zero → ping sweep or ICMP flood)",
    "stat_max":         "Max packet inter-arrival time",
    "stat_min":         "Min packet inter-arrival time",
    "stat_stddev":      "Stddev of inter-arrival times (high → irregular/malicious pattern)",
    "estado_conn":      "Connection state / TCP flags bitmask",
}

PROMPT_CLASIFICACION = """You are a network security analyst. A flow has already been flagged as \
anomalous by the detection system. Your task is to determine the ATTACK TYPE.

Classify it into exactly ONE of these categories:
{categorias}

Feature legend (name: semantic meaning):
{feature_legend}

Observed flow features:
{packet_json}

Rules:
- Do NOT output "Normal" — this flow is confirmed anomalous.
- Respond with ONLY the category name. No explanation, no punctuation.
- Use feature values AND their semantic meaning to decide.

Attack category:"""


def taxonomia_gt(nombre: str) -> str:
    """Mapea una etiqueta raw del dataset a la TAXONOMIA_LLM.

    Devuelve 'Unknown' para etiquetas no reconocidas en vez de forzar
    una categoría incorrecta, evitando contaminar el ground truth.
    """
    key = str(nombre).strip().lower()
    return TAXONOMIA_LLM.get(key, "Unknown")


def normalizar_pred_llm(respuesta: str) -> str:
    """Normaliza la respuesta del LLM a una categoria valida."""
    resp = respuesta.strip()
    # Exacto
    for c in CATEGORIAS_LLM:
        if c.lower() == resp.lower():
            return c
    # Contenido
    resp_lower = resp.lower()
    if "ddos" in resp_lower or "dos" in resp_lower or "flood" in resp_lower:
        return "DDoS/DoS"
    if "scan" in resp_lower or "recon" in resp_lower or "fingerprint" in resp_lower:
        return "Scanning"
    if "inject" in resp_lower or "sql" in resp_lower or "xss" in resp_lower:
        return "Injection"
    if "malware" in resp_lower or "backdoor" in resp_lower or "ransomware" in resp_lower:
        return "Malware"
    if "credential" in resp_lower or "mitm" in resp_lower or "password" in resp_lower:
        return "Credential/MITM"
    if "normal" in resp_lower or "benign" in resp_lower:
        return "Normal"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 1: Clasificacion Binaria (Fallback XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_binario_universal() -> dict:
    """
    Evalua el modelo Fallback sobre el Combined Test Set de los 3 datasets.
    Usa el MAPA_EQUIVALENCIAS oficial para extraer las 12 features canonicas.
    """
    logger.info(f"\n{'='*60}")
    logger.info("PARTE 1: Clasificacion Binaria (Fallback XGBoost Universal)")

    if not Path(MODELO_FALLBACK).exists():
        logger.error("Modelo fallback no encontrado.")
        return {}

    frames_X, frames_y = [], []

    for ds_id, config in DATASETS_CONFIG.items():
        if ds_id == "Autodetección" or not config.get("csv"):
            continue
        csv_path = Path(config["csv"])
        if not csv_path.exists():
            continue

        X_ds = extraer_features_dataset(csv_path, ds_id, FEATURES_COMUNES)
        label_col = LABEL_COLS[ds_id][1]
        df_lbl = pd.read_csv(csv_path, usecols=[label_col], low_memory=False)
        y_ds = df_lbl[label_col].astype(str).apply(normalizar_clase)

        frames_X.append(X_ds)
        frames_y.append(y_ds)

    X_all = pd.concat(frames_X, ignore_index=True).fillna(0.0)
    y_all = pd.concat(frames_y, ignore_index=True)

    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )
    if len(X_test) > N_BIN_SAMPLES:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), N_BIN_SAMPLES, replace=False)
        X_test = X_test.iloc[idx]
        y_test  = y_test.iloc[idx]

    logger.info(f"  Test set combinado: {len(X_test):,} muestras | dist: {dict(y_test.value_counts())}")

    centinela = AgenteCentinela(modelo_path=str(MODELO_FALLBACK))
    y_test_num = centinela.label_encoder.transform(y_test)
    y_pred     = centinela.modelo.predict(X_test[FEATURES_COMUNES].values)

    # Latencia de inferencia
    tiempos = []
    for i in range(min(500, len(X_test))):
        t0 = time.perf_counter()
        centinela.predecir(X_test.iloc[i:i+1][FEATURES_COMUNES].values)
        t1 = time.perf_counter()
        tiempos.append((t1 - t0) * 1_000_000)  # µs

    lat_media = round(float(np.mean(tiempos)), 1)
    lat_p95   = round(float(np.percentile(tiempos, 95)), 1)

    acc  = accuracy_score(y_test_num, y_pred)
    f1_w = f1_score(y_test_num, y_pred, average='weighted')
    f1_b = f1_score(y_test_num, y_pred, average='macro')

    logger.info(f"  Accuracy:        {acc:.4f}")
    logger.info(f"  F1-Weighted:     {f1_w:.4f}")
    logger.info(f"  F1-Macro:        {f1_b:.4f}")
    logger.info(f"  Latencia:        {lat_media:.0f} µs/paquete (P95: {lat_p95:.0f} µs)")

    return {
        "n_eval":         len(X_test),
        "accuracy":       round(acc, 4),
        "f1_weighted":    round(f1_w, 4),
        "f1_macro":       round(f1_b, 4),
        "latencia_us":    {"media": lat_media, "p95": lat_p95},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 2: Clasificacion Multiclase del LLM Analista
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_llm_multiclase(dataset_id: str, csv_path: Path) -> dict | None:
    """
    Evalúa la capacidad del LLM Analista para clasificar ataques en la taxonomía
    semántica de 6 categorías usando el PROMPT_CLASIFICACION restringido.

    A diferencia de la firma libre del orquestador (analista.analizar()), este
    benchmark utiliza directamente la Groq API con el prompt constreñido a las
    6 categorías de CATEGORIAS_LLM, replicando la evaluación de un sistema de
    clasificación semántica de amenazas IoT.

    El ground truth se mapea mediante TAXONOMIA_LLM. Solo se evalúan muestras
    con etiqueta GT reconocida (no 'Unknown') para evitar contaminar las métricas.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning(f"  [{dataset_id}] GROQ_API_KEY no encontrada. Saltando LLM.")
        return None

    label_col = LABEL_COLS[dataset_id][1]
    df = pd.read_csv(csv_path, low_memory=False)

    if label_col not in df.columns:
        logger.warning(f"  [{dataset_id}] Columna '{label_col}' no encontrada.")
        return None

    # Mapear etiquetas GT a la taxonomía semántica; descartar los 'Unknown'
    df["_gt_raw"]  = df[label_col].astype(str)
    df["_gt_taxo"] = df["_gt_raw"].apply(taxonomia_gt)

    # Excluir Normal y etiquetas no reconocidas del GT
    df_ataques = df[(df["_gt_taxo"] != "Normal") & (df["_gt_taxo"] != "Unknown")].copy()

    if len(df_ataques) == 0:
        logger.warning(f"  [{dataset_id}] Sin muestras de ataque con GT reconocido.")
        return None

    logger.info(f"  [{dataset_id}] Clases GT disponibles: {sorted(df_ataques['_gt_taxo'].unique())}")

    # Muestra estratificada por categoría semántica
    muestras = []
    categorias_presentes = df_ataques["_gt_taxo"].unique()
    por_clase = max(1, N_LLM_SAMPLES // len(categorias_presentes))
    for cat in categorias_presentes:
        sub = df_ataques[df_ataques["_gt_taxo"] == cat]
        muestras.append(sub.sample(n=min(por_clase, len(sub)), random_state=RANDOM_STATE))
    df_muestra = pd.concat(muestras).sample(frac=1, random_state=RANDOM_STATE).head(N_LLM_SAMPLES)

    # Eliminar columnas de etiqueta para no revelar la respuesta al LLM
    cols_borrar = [c for c in LABEL_COLS[dataset_id] if c in df_muestra.columns] + ["_gt_raw", "_gt_taxo"]
    df_input = df_muestra.drop(columns=cols_borrar, errors="ignore")

    # Cliente Groq directo — prompt constreñido a CATEGORIAS_LLM
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        logger.warning(f"  [{dataset_id}] No se pudo instanciar Groq: {e}")
        return None

    categorias_ataque = [c for c in CATEGORIAS_LLM if c != "Normal"]  # ya sabemos que es ataque
    categorias_str = "\n".join(f"- {c}" for c in categorias_ataque)
    feature_legend = "\n".join(
        f"  {k}: {v}" for k, v in FEATURE_DESCRIPTIONS.items()
    )
    y_true  = df_muestra["_gt_taxo"].tolist()
    y_pred  = []
    tiempos = []

    logger.info(f"  [{dataset_id}] Evaluando {len(df_muestra)} muestras con prompt constreñido...")

    for idx_row, (_, row) in enumerate(df_input.iterrows()):
        # Serializar el paquete eliminando NaNs
        paquete = {k: v for k, v in row.to_dict().items() if not pd.isna(v)}
        # Limitar a 40 features para no exceder la ventana de contexto
        if len(paquete) > 40:
            paquete = dict(list(paquete.items())[:40])
        paquete_str = json.dumps(paquete, ensure_ascii=False)

        prompt = PROMPT_CLASIFICACION.format(
            categorias=categorias_str,
            feature_legend=feature_legend,
            packet_json=paquete_str
        )

        t0 = time.perf_counter()
        try:
            respuesta = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODELO_ANALISTA,
                temperature=0.0,   # determinístico
                max_tokens=30,     # suficiente para 'Credential/MITM'
                timeout=30
            )
            t1 = time.perf_counter()
            tiempos.append((t1 - t0) * 1000)

            texto_respuesta = respuesta.choices[0].message.content
            pred = normalizar_pred_llm(texto_respuesta)
            y_pred.append(pred)
            logger.info(
                f"    [{idx_row+1:02d}/{len(df_muestra)}] "
                f"GT={y_true[idx_row]:<20} → raw='{texto_respuesta[:25]}' → Norm={pred}"
            )

        except Exception as e:
            logger.warning(f"    Error LLM muestra {idx_row}: {e}")
            y_pred.append("Unknown")
            time.sleep(3)

        time.sleep(0.5)  # respetar rate-limit

    # Calcular métricas excluyendo respuestas Unknown
    validos = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
    if not validos:
        logger.warning(f"  [{dataset_id}] Todas las predicciones son Unknown. Sin métricas.")
        return None

    yt = [v[0] for v in validos]
    yp = [v[1] for v in validos]
    pct_unknown = (len(y_true) - len(yt)) / len(y_true) * 100

    labels_presentes = sorted(set(yt + yp) - {"Unknown"})
    f1_m = f1_score(yt, yp, average="macro",    labels=labels_presentes, zero_division=0)
    f1_w = f1_score(yt, yp, average="weighted", labels=labels_presentes, zero_division=0)
    acc  = accuracy_score(yt, yp)

    arr = np.array(tiempos) if tiempos else np.array([0])
    lat = {"media_ms": round(float(arr.mean()), 0), "p95_ms": round(float(np.percentile(arr, 95)), 0)}

    logger.info(
        f"  [{dataset_id}] F1-Macro={f1_m:.4f}  F1-W={f1_w:.4f}  "
        f"Acc={acc:.4f}  Unknown={pct_unknown:.0f}%  Lat={lat['media_ms']:.0f}ms"
    )

    return {
        "n_eval":      len(y_true),
        "n_validos":   len(yt),
        "pct_unknown": round(pct_unknown, 1),
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "latencia_ms": lat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("BENCHMARK FASE 1 — Survival Mode (XGBoost Fallback + LLM Analista)")

    if not FEATURES_COMUNES:
        logger.error("FEATURES_COMUNES no cargadas. Ejecuta: python src/generar_features_comunes.py")
        return

    resultado_final = {}

    # ── Parte 1: Binario Universal ────────────────────────────────────────────
    resultado_final["binario_universal"] = evaluar_binario_universal()

    # ── Parte 2: LLM Multiclase por dataset ──────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PARTE 2: Clasificacion Multiclase (LLM Analista)")
    logger.info(f"Taxonomia evaluada: {CATEGORIAS_LLM}")

    resultado_final["llm_multiclase"] = {}

    for ds_id, config in DATASETS_CONFIG.items():
        if ds_id == "Autodetección" or not config.get("csv"):
            continue
        csv_path = Path(config["csv"])
        if not csv_path.exists():
            continue

        res_llm = evaluar_llm_multiclase(ds_id, csv_path)
        resultado_final["llm_multiclase"][ds_id] = res_llm

    # ── Resumen ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN FINAL — Fase 1 (Survival Mode)")
    b = resultado_final["binario_universal"]
    logger.info(f"\n  [Binario Universal - Fallback XGBoost]")
    logger.info(f"    Accuracy: {b.get('accuracy'):.4f}  F1-W: {b.get('f1_weighted'):.4f}  Lat: {b.get('latencia_us',{}).get('media','?')} µs")

    logger.info(f"\n  [Multiclase - LLM Analista (Taxonomia: {len(CATEGORIAS_LLM)} categorias)]")
    for ds_id, res in resultado_final["llm_multiclase"].items():
        if res:
            logger.info(f"    {ds_id:<12}: F1-Macro={res['f1_macro']:.4f}  F1-W={res['f1_weighted']:.4f}  Lat={res['latencia_ms']['media_ms']:.0f}ms")

    # ── Guardar ───────────────────────────────────────────────────────────────
    ruta = RESULTADOS_DIR / "benchmark_fase1.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados guardados en: {ruta}")


if __name__ == "__main__":
    main()
