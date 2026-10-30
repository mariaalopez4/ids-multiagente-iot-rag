"""Capa de reporting y visualizacion para metricas de clasificacion.

Consume metricas calculadas por AgenteCentinela y genera reportes academicos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class GeneradorMetricas:
    """Genera visualizaciones y reportes academicos a partir de metricas calculadas."""

    def __init__(self, agente_centinela, directorio_salida: str = "./resultados"):
        """
        Args:
            agente_centinela: Instancia entrenada de AgenteCentinela
            directorio_salida: Directorio donde guardar los reportes
        """
        self.agente = agente_centinela
        self.dir_salida = Path(directorio_salida)
        self.dir_salida.mkdir(parents=True, exist_ok=True)
        
        if not self.agente.esta_entrenado:
            raise RuntimeError("El agente debe estar entrenado primero")
        
        self.metricas = self.agente.metricas
        self.clases = list(self.agente.label_encoder.classes_)
    
    def generar_matriz_confusion(self, normalizar: bool = True, figsize=(12, 10)):
        """Genera y guarda matriz de confusion como PNG."""
        logger.info("Generando matriz de confusion...")
        
        cm = np.array(self.metricas['confusion_matrix'])
        
        if normalizar:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            formato = '.2%'
            titulo = 'Matriz de Confusión Normalizada'
        else:
            formato = 'd'
            titulo = 'Matriz de Confusión'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True,  
            fmt=formato, 
            cmap='Blues',
            xticklabels=self.clases,
            yticklabels=self.clases,
            cbar_kws={'label': 'Proporción' if normalizar else 'Conteo'},
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
        ax.set_ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        ruta = self.dir_salida / 'confusion_matrix.png'
        plt.savefig(ruta, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Guardada en {ruta}")
        return str(ruta)
    
    def generar_reporte_clasificacion(self):
        """Exporta reporte de clasificacion a CSV."""
        logger.info("Generando reporte de clasificacion...")
        
        reporte = self.metricas['reporte_clases']
        df_reporte = pd.DataFrame(reporte).transpose()
        
        for col in ['precision', 'recall', 'f1-score']:
            if col in df_reporte.columns:
                df_reporte[col] = df_reporte[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, float) else x
                )
        
        if 'support' in df_reporte.columns:
            df_reporte['support'] = df_reporte['support'].apply(
                lambda x: int(x) if isinstance(x, (int, float)) else x
            )
        
        ruta = self.dir_salida / 'classification_report.csv'
        df_reporte.to_csv(ruta, index=True, encoding='utf-8-sig')
        logger.info(f"Guardado en {ruta}")
        
        self._mostrar_top_clases(df_reporte)
        return str(ruta)
    
    def _mostrar_top_clases(self, df_reporte):
        """Helper: muestra top 5 clases por F1-Score."""
        df_temp = df_reporte[df_reporte.index.isin(self.agente.ATAQUES_CONOCIDOS)].copy()
        if 'f1-score' in df_temp.columns:
            df_temp['f1-score'] = df_temp['f1-score'].astype(float)
            top5 = df_temp.nlargest(5, 'f1-score')[['f1-score', 'support']]
            logger.info("Top 5 por F1-Score:")
            logger.info(top5.to_string())
    
    def generar_importancia_features(self, top_n: int = 20, figsize=(10, 8)):
        """Genera grafico de importancia de features."""
        logger.info(f"Generando importancia de features (Top {top_n})...")
        
        importancias_dict = self.agente.obtener_importancia_features(top_n)
        df_imp = pd.DataFrame(
            list(importancias_dict.items()),
            columns=['Feature', 'Importancia']
        ).sort_values('Importancia', ascending=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df_imp['Feature'], df_imp['Importancia'],
                color=sns.color_palette("viridis", len(df_imp)))
        
        ax.set_xlabel('Importancia (Gain)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features Más Importantes', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        ruta = self.dir_salida / 'feature_importance.png'
        plt.savefig(ruta, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Guardado en {ruta}")
        return str(ruta)
    
    def generar_resumen_metricas(self):
        """Genera dashboard visual con metricas globales y por clase."""
        logger.info("Generando resumen de metricas...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        self._graficar_metricas_globales(axes[0])
        self._graficar_f1_por_clase(axes[1])
        
        plt.tight_layout()
        
        ruta = self.dir_salida / 'resumen_metricas.png'
        plt.savefig(ruta, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Guardado en {ruta}")
        return str(ruta)
    
    def _graficar_metricas_globales(self, ax):
        """Helper: grafica accuracy, f1, precision, recall."""
        metricas_globales = {
            'Accuracy': self.metricas['accuracy'],
            'F1 (Macro)': self.metricas['f1_macro'],
            'Precision (Macro)': self.metricas['precision_macro'],
            'Recall (Macro)': self.metricas['recall_macro']
        }
        
        colores = sns.color_palette("Set2", len(metricas_globales))
        bars = ax.bar(metricas_globales.keys(), metricas_globales.values(),
                     color=colores, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Métricas Globales', fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    def _graficar_f1_por_clase(self, ax):
        """Helper: grafica F1-Score por tipo de ataque."""
        reporte = self.metricas['reporte_clases']
        clases_f1 = {
            clase: datos['f1-score'] 
            for clase, datos in reporte.items() 
            if clase in self.agente.ATAQUES_CONOCIDOS
        }
        
        clases_ordenadas = sorted(clases_f1.items(), key=lambda x: x[1], reverse=True)
        clases, f1_scores = zip(*clases_ordenadas)
        colores = ['green' if f1 >= 0.9 else 'orange' if f1 >= 0.7 else 'red' 
                  for f1 in f1_scores]
        
        ax.barh(clases, f1_scores, color=colores, edgecolor='black', linewidth=0.8)
        ax.set_xlabel('F1-Score', fontweight='bold')
        ax.set_title('F1-Score por Ataque', fontweight='bold', pad=15)
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Excelente (≥0.9)'),
            Patch(facecolor='orange', label='Bueno (0.7-0.9)'),
            Patch(facecolor='red', label='Mejorable (<0.7)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    def exportar_json_completo(self):
        """Exporta todas las metricas a JSON."""
        logger.info("Exportando metricas a JSON...")
        ruta = self.dir_salida / 'metricas_completas.json'
        self.agente.exportar_metricas(str(ruta))
        return str(ruta)
    
    def generar_todo(self):
        """Genera todos los reportes: matriz, CSV, features, dashboard y JSON."""
        logger.info("Generando metricas...")
        archivos = {}
        
        try:
            archivos['confusion_matrix'] = self.generar_matriz_confusion()
            archivos['reporte_csv'] = self.generar_reporte_clasificacion()
            archivos['feature_importance'] = self.generar_importancia_features()
            archivos['resumen_visual'] = self.generar_resumen_metricas()
            archivos['json_completo'] = self.exportar_json_completo()
            
            self._crear_readme(archivos)
            
            logger.info(f"Completado. Archivos en: {self.dir_salida}")
            return archivos
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def _crear_readme(self, archivos):
        """Helper: crea README.md con resumen de archivos generados."""
        readme_path = self.dir_salida / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Metricas del Agente Centinela\n\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Archivos\n\n")
            for nombre, ruta in archivos.items():
                f.write(f"- **{nombre}**: `{Path(ruta).name}`\n")


if __name__ == "__main__":
    from agente_centinela import AgenteCentinela
    from config import MODELO_CENTINELA, RESULTADOS_DIR
    
    print("Cargando agente entrenado...")
    try:
        agente = AgenteCentinela(modelo_path=str(MODELO_CENTINELA))
        generador = GeneradorMetricas(agente, directorio_salida=str(RESULTADOS_DIR))
        archivos = generador.generar_todo()
        print("\nArchivos generados:")
        for nombre, ruta in archivos.items():
            print(f"  - {nombre}: {ruta}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"\nAsegúrate de que el modelo está en: {MODELO_CENTINELA}")
    

