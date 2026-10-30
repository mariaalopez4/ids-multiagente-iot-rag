"""Componentes visuales reutilizables para la interfaz Streamlit."""

import streamlit as st
from pathlib import Path

def load_css(file_name="assets/style.css"):
    """Carga un archivo CSS externo en la aplicación Streamlit."""
    css_path = Path(__file__).parent / file_name
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error cargando CSS: {e}")

def badge_html(texto, tipo="critical"):
    """Genera el HTML para un badge de estado."""
    return f'<span class="badge-{tipo}">{texto}</span>'

def kpi_card(valor, etiqueta):
    """Genera el HTML para una tarjeta de KPI."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{valor}</div>
        <div class="kpi-label">{etiqueta}</div>
    </div>
    """, unsafe_allow_html=True)

def agent_card_header(titulo, subtitulo=None):
    """Retorna el inicio de una tarjeta de agente."""
    html = f'<div class="agent-card"><h4>{titulo}</h4>'
    if subtitulo:
        html += f'<div class="section-title">{subtitulo}</div>'
    return html

def color_severidad(sev):
    """Retorna el color hexadecimal asociado a una severidad."""
    mapa = {
        "Crítica": "#f85149",
        "Alta": "#f0883e",
        "Media": "#d29922",
        "Baja": "#3fb950",
    }
    for k, v in mapa.items():
        if k.lower() in str(sev).lower():
            return v
    return "#8b949e"

def learning_indicator():
    """Muestra un indicador discreto de aprendizaje en segundo plano."""
    st.sidebar.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; padding: 10px; 
                    background: #1a2233; border-radius: 8px; border: 1px solid #388bfd33;
                    margin-bottom: 20px;'>
            <div class="spinner-mini"></div>
            <span style='font-size: 0.8rem; color: #79c0ff;'>
                Fase 3: Entrenando modelo experto en background...
            </span>
        </div>
        <style>
            .spinner-mini {
                width: 12px; height: 12px;
                border: 2px solid #388bfd33;
                border-top: 2px solid #388bfd;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
        """, unsafe_allow_html=True
    )

def centinela_card(tipo_ataque, confianza, es_normal, dataset_origen, fallback_activo):
    """Genera el panel visual del Agente Centinela."""
    color_tipo = "#3fb950" if es_normal else "#f85149"
    badge_tipo = badge_html("Tráfico legítimo", "ok") if es_normal else badge_html("Amenaza detectada", "critical")
    
    st.markdown(f"""
    <div class="agent-card">
        <h4>👁 Centinela — XGBoost</h4>
        {badge_tipo}
        <div class="{'ok-value' if es_normal else 'threat-value'}">{tipo_ataque}</div>
        <div class="section-title">Confianza</div>
        <div style="font-size:1.1rem; font-weight:600; color:{color_tipo};">{confianza:.1%}</div>
        <div class="section-title" style="margin-top:10px;">Origen</div>
        <div style="font-size:0.75rem; color:#8b949e;">{dataset_origen}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if fallback_activo:
        st.markdown("<div style='margin-top:10px; padding:5px; background-color:#3d2011; border:1px solid #f0883e; border-radius:5px; color:#f0883e; font-size:0.8rem; text-align:center;'><b>⚠️ MODO SUPERVIVENCIA</b><br>Fallback LLM Activo (Degradado)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:10px; padding:5px; background-color:#163b21; border:1px solid #3fb950; border-radius:5px; color:#3fb950; font-size:0.8rem; text-align:center;'><b>✅ MODO EXPERTO</b><br>Alta Disponibilidad (Latencia Óptima)</div>", unsafe_allow_html=True)

def format_md_to_html(text):
    """Convierte markdown básico a HTML para renderizarlo correctamente dentro de los divs."""
    if not text: return "—"
    import re
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Code blocks simple
    text = re.sub(r'`(.*?)`', r'<code style="background-color:#161b22; padding:2px 4px; border-radius:3px;">\1</code>', text)
    
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('* ') or line.startswith('- '):
            if not in_list:
                formatted_lines.append('<ul style="margin-top: 5px; margin-bottom: 5px; padding-left: 20px; list-style-type: disc;">')
                in_list = True
            formatted_lines.append(f'<li style="margin-bottom: 3px;">{line[2:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            if line:
                formatted_lines.append(f'<div style="margin-bottom: 6px;">{line}</div>')
    
    if in_list:
        formatted_lines.append('</ul>')
        
    return ''.join(formatted_lines)

def analista_card(analisis, es_normal, t_total):
    """Genera el panel visual del Agente Analista."""
    if analisis:
        sev = analisis.get("severidad", "Media")
        col_sev = color_severidad(sev)
        explicacion_html = format_md_to_html(analisis.get('explicacion', '—'))
        st.markdown(f"""
        <div class="agent-card">
            <h4>🧠 Analista — LLM</h4>
            <div class="section-title">Severidad</div>
            <div style="font-size:1.1rem; font-weight:700; color:{col_sev}; margin-bottom:8px;">{sev}</div>
            <div class="section-title">Latencia Análisis</div>
            <div style="font-size:0.85rem; color:#e6edf3; margin-bottom:10px;">{t_total:.3f}s</div>
            <div class="section-title">Análisis</div>
            <div style="font-size:0.82rem; color:#c9d1d9; line-height:1.5;">{explicacion_html}</div>
        </div>
        """, unsafe_allow_html=True)
    elif not es_normal:
        st.markdown(f"""
        <div class="agent-card">
            <h4>🧠 Analista — LLM</h4>
            <div class="badge-critical">⚠ Análisis no disponible</div>
            <p style="color:#8b949e; font-size:0.85rem; margin-top:10px;">
                El agente analista no pudo procesar esta amenaza.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="agent-card">
            <h4>🧠 Analista — LLM</h4>
            <p style="color:#8b949e; font-size:0.85rem;">No se requiere análisis para tráfico legítimo.</p>
        </div>
        """, unsafe_allow_html=True)

def respondedor_card(respuesta, fallback_activo, t_norm_llm=0):
    """Genera el panel visual del Agente Respondedor."""
    if respuesta:
        mitig = respuesta.get("mitigacion_tecnica", "Sin datos")
        mitig_html = format_md_to_html(mitig)
        st.markdown(f"""
        <div class="agent-card">
            <h4>🛡 Respondedor — RAG</h4>
            <div class="section-title">Plan de respuesta activa</div>
            <div style="font-size:0.85rem; color:#c9d1d9; line-height:1.6; margin-top:10px;">
                {mitig_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if fallback_activo:
            st.info(f"⚡ Mapeo dinámico realizado (+{round(t_norm_llm, 3)}s)")
    else:
        st.markdown("""
        <div class="agent-card">
            <h4>🛡 Respondedor — RAG</h4>
            <p style="color:#8b949e; font-size:0.85rem;">Sin acciones necesarias para tráfico normal.</p>
        </div>
        """, unsafe_allow_html=True)
