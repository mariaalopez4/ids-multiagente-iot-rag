"""Agente Analista para razonamiento LLM sobre alertas de seguridad."""

import os
import logging
import time
from typing import Optional, Dict
from dataclasses import dataclass

from groq import Groq
from config import MODELO_ANALISTA, MODELO_RESPONDEDOR

logger = logging.getLogger(__name__)


@dataclass
class ResultadoAnalisis:
    tipo_ataque: str
    explicacion: str
    tactica_mitre: str
    severidad: str
    justificacion_severidad: str
    tiempo_procesamiento: float
    modelo_usado: str


class AgenteAnalista:
    PROMPT_SISTEMA = """Eres un analista de ciberseguridad.
Analiza logs de ataques IoT y genera reportes estructurados.

Debes identificar el tipo de ataque, explicar su mecanica, mapear a MITRE ATT&CK y evaluar severidad.
Si el 'Tipo' provisto es 'Ataque' o genérico, DEDUCE tú mismo la taxonomía exacta (ej. SQL Injection, Port Scanning, MITM) leyendo el JSON del 'Contexto'.

Formato de salida:
### TIPO DE ATAQUE: [Nombre]
### ANALISIS TECNICO: [Explicacion concisa]
### MITRE ATT&CK: [Tactica/Tecnica con ID]
### SEVERIDAD: [Nivel] - [Justificacion]

Se preciso y tecnico."""
    
    # Template para consultas del usuario
    TEMPLATE_CONSULTA = """Ataque detectado en red IoT:

Tipo: {tipo_ataque}
Contexto: {contexto}

Analiza segun el formato establecido."""
    
    def __init__(self, api_key: Optional[str] = None, modelo: str = MODELO_ANALISTA,
                 temperatura: float = 0.1, max_reintentos: int = 3, timeout: int = 30):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere API Key de Groq en GROQ_API_KEY")
        
        self.modelo = modelo
        self.temperatura = temperatura
        self.max_reintentos = max_reintentos
        self.timeout = timeout
        
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info("Cliente Groq conectado correctamente")
        except Exception as e:
            logger.error(f"Error al conectar con Groq: {e}")
            raise
    
    def analizar(self, tipo_ataque: str, contexto: Optional[Dict] = None) -> ResultadoAnalisis:
        inicio = time.time()
        if not tipo_ataque or not isinstance(tipo_ataque, str):
            raise ValueError("tipo_ataque debe ser un string")
        
        contexto_str = self._preparar_contexto(tipo_ataque, contexto or {})
        prompt_usuario = self.TEMPLATE_CONSULTA.format(tipo_ataque=tipo_ataque, contexto=contexto_str)
        respuesta_texto = self._llamar_llm_con_reintentos(prompt_usuario, tipo_ataque)
        resultado = self._parsear_respuesta(respuesta_texto, tipo_ataque, time.time() - inicio)
        
        logger.info(f"Analisis de {tipo_ataque} completado en {resultado.tiempo_procesamiento:.2f}s")
        return resultado
    
    def _preparar_contexto(self, tipo_ataque: str, contexto: Dict) -> str:
        partes = [f"Ataque detectado: {tipo_ataque}"]
        
        if 'src_ip' in contexto:
            partes.append(f"IP origen: {contexto['src_ip']}")
        if 'dst_port' in contexto:
            partes.append(f"Puerto destino: {contexto['dst_port']}")
        if 'protocol' in contexto:
            partes.append(f"Protocolo: {contexto['protocol']}")
        if 'timestamp' in contexto:
            partes.append(f"Timestamp: {contexto['timestamp']}")
        
        for key, value in contexto.items():
            if key not in ['src_ip', 'dst_port', 'protocol', 'timestamp']:
                partes.append(f"{key}: {value}")
        
        return "; ".join(partes)
    
    def _llamar_llm_con_reintentos(self, prompt_usuario: str, tipo_ataque: str) -> str:
        ultimo_error = None
        modelo_actual = self.modelo
        
        for intento in range(self.max_reintentos):
            try:
                respuesta = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.PROMPT_SISTEMA},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    model=modelo_actual,
                    temperature=self.temperatura,
                    timeout=self.timeout
                )
                return respuesta.choices[0].message.content
            except Exception as e:
                ultimo_error = e
                # Si es error de Rate Limit (429) y estamos usando el 70B, 
                # bajamos inmediatamente al 8B para intentar salvar la respuesta.
                if "rate_limit_exceeded" in str(e).lower() and "70b" in modelo_actual:
                    logger.warning(f"Límite de Groq 70B alcanzado. Aplicando FALLBACK a modelo 8B...")
                    modelo_actual = MODELO_RESPONDEDOR
                    continue # Reintento inmediato con el modelo ligero
                
                if intento < self.max_reintentos - 1:
                    time.sleep(2 ** intento)
        
        return self._generar_fallback(tipo_ataque, ultimo_error)
    
    def _generar_fallback(self, tipo_ataque: str, error: Exception) -> str:
        return f"""### TIPO DE ATAQUE: {tipo_ataque}
### ANALISIS TECNICO: No disponible
### MITRE ATT&CK: Pendiente
### SEVERIDAD: Media - Requiere revision manual"""
    
    def _parsear_respuesta(
        self, 
        respuesta_texto: str, 
        tipo_ataque: str,
        tiempo: float
    ) -> ResultadoAnalisis:
        """Extrae la información del formato estructurado del LLM de forma robusta y multilínea."""
        analisis = {
            "tipo_ataque": tipo_ataque,
            "explicacion": "Análisis no disponible",
            "tactica_mitre": "—",
            "severidad": "Media",
            "justificacion": "Análisis automático"
        }
        
        # Mapeo de cabeceras para el parser
        mapeo = {
            "TIPO DE ATAQUE:": "tipo_ataque",
            "ANALISIS TECNICO:": "explicacion",
            "MITRE ATT&CK:": "tactica_mitre",
            "SEVERIDAD:": "severidad"
        }
        
        lineas = respuesta_texto.split('\n')
        seccion_actual = None
        contenido_seccion = []
        
        def guardar_seccion():
            if seccion_actual and contenido_seccion:
                texto_seccion = " ".join(contenido_seccion).strip()
                if seccion_actual == "severidad" and "-" in texto_seccion:
                    parts = texto_seccion.split("-", 1)
                    analisis["severidad"] = parts[0].strip()
                    analisis["justificacion"] = parts[1].strip()
                else:
                    analisis[seccion_actual] = texto_seccion

        for linea in lineas:
            linea_limpia = linea.strip()
            if not linea_limpia: continue
            
            # Detectar si la línea es una cabecera
            nueva_seccion = None
            for key, campo in mapeo.items():
                if key in linea_limpia.upper() or linea_limpia.upper().startswith(key.replace("### ", "")):
                    nueva_seccion = campo
                    break
            
            if nueva_seccion:
                guardar_seccion()
                seccion_actual = nueva_seccion
                # Extraer posible contenido en la misma línea que el encabezado
                content = linea.split(":", 1)[-1].replace("]", "").strip()
                contenido_seccion = [content] if content else []
            elif seccion_actual:
                # Acumular líneas de contenido
                txt = linea_limpia.replace("###", "").strip()
                if txt:
                    contenido_seccion.append(txt)
        
        guardar_seccion() # Guardar la última

        return ResultadoAnalisis(
            tipo_ataque=analisis["tipo_ataque"],
            explicacion=analisis["explicacion"],
            tactica_mitre=analisis["tactica_mitre"],
            severidad=analisis["severidad"],
            justificacion_severidad=analisis["justificacion"],
            tiempo_procesamiento=tiempo,
            modelo_usado=self.modelo
        )
    
    def generar_informe_texto(self, resultado: ResultadoAnalisis) -> str:
        return f"""ANALISIS DE ATAQUE
---
Tipo: {resultado.tipo_ataque}
Analisis: {resultado.explicacion}
MITRE: {resultado.tactica_mitre}
Severidad: {resultado.severidad}
Justificacion: {resultado.justificacion_severidad}
Tiempo: {resultado.tiempo_procesamiento:.2f}s
Modelo: {resultado.modelo_usado}"""


if __name__ == "__main__":
    print("Agente Analista - Modulo cargado correctamente")
