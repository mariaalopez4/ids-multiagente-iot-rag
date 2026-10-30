"""Sistema RAG para mitigaciones basadas en MITRE CAPEC."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import threading
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import MODELO_RESPONDEDOR

logger = logging.getLogger(__name__)


class AgenteRespondedor:
    """Sistema RAG para consultar mitigaciones MITRE CAPEC usando FAISS y LangChain."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        modelo_llm: str = MODELO_RESPONDEDOR,
        modelo_embedding: str = "sentence-transformers/all-MiniLM-L6-v2",
        ruta_kb: Optional[str] = None,
        ruta_indices: Optional[str] = None
    ):
        """Inicializa el agente RAG con modelos de embedding y LLM."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere API Key de Groq")
        
        os.environ["GROQ_API_KEY"] = self.api_key
        
        self.modelo_llm = modelo_llm
        self.modelo_embedding = modelo_embedding
        self.ruta_indices = Path(ruta_indices) if ruta_indices else None
        
        self.vectorstore = None
        self.qa_chain = None
        
        self.indice_listo = False
        self._hilo_carga = threading.Thread(
            target=self.construir_indice, 
            args=(ruta_kb,), 
            daemon=True
        )
        self._hilo_carga.start()
    
    def construir_indice(self, ruta_kb: Optional[str] = None):
        logger.info("Construyendo índice vectorial...")
        
        if self.ruta_indices and self.ruta_indices.exists():
            try:
                logger.info(f"Cargando índice desde cache: {self.ruta_indices}")
                
                # Intentar cargar embeddings (con manejo de fallos de red)
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name=self.modelo_embedding,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                except Exception as e_emb:
                    logger.error(f"Error de red al cargar Embeddings (HuggingFace puede estar caído): {e_emb}")
                    logger.warning("El Agente Respondedor NO estará disponible, pero el IDS seguirá funcionando.")
                    self.indice_listo = False
                    return

                self.vectorstore = FAISS.load_local(
                    str(self.ruta_indices),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self._construir_cadena_qa()
                self.indice_listo = True
                logger.info("Índice cargado desde cache")
                return
            except Exception as e:
                logger.warning(f"No se pudo cargar cache: {e}. Reconstruyendo...")
        
        if not ruta_kb:
            # Buscar archivo por defecto en el directorio actual
            ruta_por_defecto = Path(__file__).parent / "mitre_capec_kb.txt"
            if ruta_por_defecto.exists():
                ruta_kb = str(ruta_por_defecto)
                logger.info(f"Usando KB detectada automáticamente: {ruta_kb}")
            else:
                raise FileNotFoundError(
                    "No se encontró 'mitre_capec_kb.txt'. "
                    "Por favor, asegúrate de que el archivo existe en el mismo directorio."
                )
            
        # Cargar desde archivo externo
        logger.info(f"Cargando KB desde: {ruta_kb}")
        loader = TextLoader(ruta_kb, encoding='utf-8')
        docs = loader.load()
        
        # Dividir en chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,  # Solapamiento para no perder contexto
            separator="\n\n"   # Separar por párrafos
        )
        docs_split = text_splitter.split_documents(docs)
        
        logger.info(f"Documentos divididos en {len(docs_split)} chunks")
        
        # Generar embeddings (con manejo de fallos de red)
        logger.info(f"Generando embeddings con {self.modelo_embedding}...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.modelo_embedding)
        except Exception as e_emb:
            logger.error(f"Error crítico de red en construcción de índice: {e_emb}")
            self.indice_listo = False
            return
        
        self.vectorstore = FAISS.from_documents(docs_split, embeddings)
        
        # Guardar cache si se especificó ruta
        if self.ruta_indices:
            try:
                self.ruta_indices.parent.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(self.ruta_indices))
                logger.info(f"Índice guardado en cache: {self.ruta_indices}")
            except Exception as e:
                logger.warning(f"No se pudo guardar índice en disco (ruta con caracteres especiales): {e}")
                logger.info("Índice se mantiene en memoria (no hay impacto en funcionalidad)")
        
        # Construir cadena QA
        self._construir_cadena_qa()
        
        self.indice_listo = True
        logger.info("Indice vectorial construido")
    
    def _construir_cadena_qa(self):
        """
        Construye la lógica de QA manualmente para evitar dependencias rotas.
        """
        self.llm = ChatGroq(
            model_name=self.modelo_llm,
            temperature=0,
            timeout=30
        )
        
        # Template para RAG — Respuesta Ejecutiva y Directa
        self.prompt_template = PromptTemplate(
            template="""Actúa como un Analista SOC Experto. Genera un plan de mitigación accionable basado estrictamente en la base de conocimiento MITRE CAPEC proporcionada.

            CONTEXTO TÉCNICO CAPEC:
            {context}

            AMENAZA DETECTADA: {question}
            
            Reglas de formato OBLIGATORIAS:
            1. NO uses introducciones como "Aquí tienes el plan" ni saludos.
            2. Inicia tu respuesta siempre indicando el origen CAPEC en negrita (Ej: **Mapeo MITRE CAPEC: [ID] - [Nombre de la Técnica]**).
            3. Proporciona exactamente entre 3 y 4 medidas tácticas usando viñetas (*). Usa verbos de acción (Bloquear, Aislar, Reconfigurar).
            4. El tono debe ser directo y profesional, como un informe de seguridad técnico, pero natural de leer.
            5. Si el contexto no contiene información útil para esta amenaza, indica claramente: "Se requiere análisis manual (sin medidas específicas en CAPEC para este vector)."

            Plan de Respuesta Activa:""",
            input_variables=["context", "question"]
        )
    
    def consultar(self, nombre_ataque: str) -> Dict:
        """Consulta mitigación, esperando a que el hilo de carga termine si es necesario."""
        if not self.indice_listo:
            logger.info("Esperando a que el índice vectorial termine de cargar...")
            self._hilo_carga.join(timeout=30) # Espera máxima de 30 segundos
            
        if not self.indice_listo:
            return {
                'mitigacion': "El sistema RAG aún se está inicializando. Por favor, reintenta en unos segundos.",
                'fuentes': [],
                'modelo': self.modelo_llm,
                'nombre_ataque': nombre_ataque,
                'timestamp': datetime.now().isoformat()
            }
        
        if not nombre_ataque or not isinstance(nombre_ataque, str):
            raise ValueError("nombre_ataque debe ser un string no vacío")
        
        logger.info(f"Buscando mitigación para: {nombre_ataque}")
        
        query = f"Dime el CAPEC ID y la mitigación para el ataque: {nombre_ataque}"
        
        try:
            # 1. Retrieval (Recuperación)
            docs = self.vectorstore.similarity_search(query, k=3)
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # 2. Generation (Generación)
            prompt = self.prompt_template.format(context=contexto, question=query)
            respuesta = self.llm.invoke(prompt)
            
            mitigacion_texto = respuesta.content
            
            logger.info(f"Mitigación encontrada ({len(docs)} fuentes)")
            
            return {
                'mitigacion': mitigacion_texto,
                'fuentes': docs,
                'modelo': self.modelo_llm,
                'nombre_ataque': nombre_ataque,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en consulta RAG: {e}")
            return {
                'mitigacion': f"Error en RAG: {str(e)}",
                'fuentes': [],
                'modelo': self.modelo_llm,
                'nombre_ataque': nombre_ataque,
                'timestamp': datetime.now().isoformat()
            }
    
    
    def obtener_estadisticas(self) -> Dict:
        """Retorna estadísticas del índice vectorial."""
        if not self.indice_listo:
            return {"error": "Índice no construido"}
        
        return {
            "num_documentos": self.vectorstore.index.ntotal,
            "modelo_embedding": self.modelo_embedding,
            "modelo_llm": self.modelo_llm,
            "indice_listo": self.indice_listo
        }


if __name__ == "__main__":
    print("Agente Respondedor (RAG) - Modulo cargado correctamente")
