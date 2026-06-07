# Clasificación de Marcha Patológica mediante Redes Neuronales Recurrentes

**Autor:** Weimar Andres Arenas Gonzalez  
**Email:** weimar.arenas@udea.edu.co  
**Programa:** Maestría en Ingeniería — Universidad de Antioquia  
**Curso:** Fundamentos de Deep Learning  
**Dataset:** GIST Pathological Gait Dataset (Jun et al., 2020)

---

## Video de Presentación

> ### **[▶️ VER VIDEO EN YOUTUBE](https://youtu.be/hYI-JmHZYJU)**
> 
> **Duración:** 5-10 minutos  
> **Contenido:** Exploración de datos, explicación del código, ejecución de notebooks y resultados

---

## Descripción del Proyecto

Clasificación multiclase de **6 tipos de marcha** (1 normal + 5 patológicas) a partir de secuencias temporales de coordenadas 3D de 25 articulaciones capturadas con Microsoft Kinect v2.

### Clases de Marcha

| Clase | Tipo de Marcha | Causa Clínica |
|-------|---------------|---------------|
| 0 | Normal | Marcha sana de referencia |
| 1 | Antálgica | Dolor en pie, tobillo, rodilla o cadera |
| 2 | Rígida (Stiff-legged) | Patologías articulares (artritis reumatoide) |
| 3 | Oscilante (Lurching) | Debilidad del músculo glúteo mayor |
| 4 | Steppage (Pie caído) | Debilidad del músculo tibial anterior |
| 5 | Trendelenburg | Debilidad del glúteo medio y menor |

**Problema ML:** Clasificación sequence-to-one  
**Input:** (50 frames × 75 features) — coordenadas 3D (x,y,z) de 25 articulaciones  
**Output:** Clase de marcha (0-5)

---

## Estructura del Repositorio

```
ProyectoDL20261/
│
├── 01 - Exploracion_de_datos.ipynb           # EDA: distribución, esqueleto 3D,
│                                              #   series temporales, varianza
├── 02 - Preprocesado.ipynb                   # Carga CSV, LOSO-CV, normalización,
│                                              #   selección de articulaciones
├── 03 - Arquitectura_linea_base_RNN.ipynb    # SimpleRNN 4 capas (baseline)
├── 04 - LSTM_GRU.ipynb                       # LSTM y GRU 4 capas
├── 05 - Comparacion_resultados.ipynb         # Comparación final y métricas
│
├── INFORME_PROYECTO.PDF                      # Informe ejecutivo final (7 páginas)
├── ENTREGA1.PDF                              # Informe de la Entrega 1
├── README.md                                 # Este archivo
│
├── cache/                                    # Archivos generados automáticamente
│   ├── results_rnn.pkl                       # Resultados SimpleRNN (NB03)
│   └── results_lstm_gru.pkl                  # Resultados LSTM/GRU (NB04)
│
└── figuras/                                  # Figuras generadas por notebooks
    ├── NB01/                                 # Exploración de datos
    ├── NB03/                                 # Resultados SimpleRNN
    └── NB05/                                 # Comparación de modelos
```

---

## Ejecución en Google Colab

**Todos los notebooks son directamente reproducibles en Google Colab** haciendo clic en el botón:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WeimarArenas/ProyectoDL20261/blob/main/01%20-%20Exploracion_de_datos.ipynb)

### Orden de Ejecución Recomendado

1. **01 - Exploracion_de_datos.ipynb**  
   Análisis exploratorio del dataset, distribución de clases, visualización de esqueleto 3D

2. **02 - Preprocesado.ipynb**  
   Carga de datos, splits LOSO-CV, normalización sin data leakage

3. **03 - Arquitectura_linea_base_RNN.ipynb**  
   SimpleRNN de 4 capas (línea de base) — QUICK_MODE: 10 épocas

4. **04 - LSTM_GRU.ipynb**  
   LSTM y GRU de 4 capas — QUICK_MODE: 10 épocas

5. **05 - Comparacion_resultados.ipynb**  
   Comparación de todos los modelos, gráficas y métricas clínicas

### Modos de Ejecución

Cada notebook tiene una bandera `QUICK_MODE`:

- **`QUICK_MODE = True`** (por defecto): 10 épocas, ~15 min/notebook en CPU local
- **`QUICK_MODE = False`**: 500 épocas, ~2-3h/notebook en Colab GPU (resultados completos)

---

## Dataset: GIST Pathological Gait

**Fuente:** [github.com/kooksung/pathological_gait_datasets](https://github.com/kooksung/pathological_gait_datasets)

- **Tamaño:** 7,157 secuencias
- **Dimensiones:** 50 frames × 75 features (coordenadas x,y,z de 25 articulaciones)
- **Clases:** 6 tipos de marcha (balanceadas, ~1,200 secuencias/clase)
- **Sujetos:** 10 personas (8 pacientes + 2 controles)
- **Formato:** CSV tab-separado, 101 columnas

### Cómo Obtener los Datos

Los notebooks descargan automáticamente el dataset en Colab:

```python
!git clone https://github.com/kooksung/pathological_gait_datasets.git
```

Para ejecución local:
```bash
git clone https://github.com/WeimarArenas/ProyectoDL20261.git
cd ProyectoDL20261
git clone https://github.com/kooksung/pathological_gait_datasets.git
```

---

## Arquitecturas Implementadas

### Iteración 1: SimpleRNN (Línea de Base)

**Notebook:** `03 - Arquitectura_linea_base_RNN.ipynb`

- SimpleRNN 4 capas, 125 unidades por capa
- Dos variantes: AllJoints (75 features) y LegsOnly (30 features)
- Referencia del paper: 86.23%

### Iteración 2: LSTM y GRU

**Notebook:** `04 - LSTM_GRU.ipynb`

- LSTM 4 capas, 125 unidades (puertas forget/input/output)
- GRU 4 capas, 125 unidades (puertas update/reset, 25% menos parámetros)
- Variantes AllJoints y LegsOnly
- Referencia del paper: LSTM 87.25%, GRU 90.13% (todos), GRU 93.67% (piernas)

### Configuración Común

- **Regularización:** Dropout(0.3), L2(10⁻⁴)
- **Optimizador:** Adam (η=10⁻³)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **Evaluación:** LOSO-CV (10 folds, Leave-One-Subject-Out)
- **Normalización:** StandardScaler por fold (sin data leakage)

---

## Resultados

### Resultados Obtenidos (QUICK_MODE, 10 épocas)

| Modelo | Joints | Accuracy | Desv. Std. | Notebook |
|--------|--------|----------|------------|----------|
| **Referencias del paper (Jun et al., 2020)** |
| SimpleRNN | Todos | 86.23% | - | - |
| LSTM 4L | Todos | 87.25% | - | - |
| GRU 4L | Todos | 90.13% | - | - |
| **GRU 4L ★** | **Piernas** | **93.67%** | - | - |
| **Resultados propios (QUICK_MODE)** |
| SimpleRNN | Todos | 68.29% | ±12.47% | NB03 |
| SimpleRNN | Piernas | 65.10% | ±8.08% | NB03 |
| LSTM 4L | Todos | 80.35% | ±14.53% | NB04 |
| GRU 4L | Todos | 75.56% | ±15.81% | NB04 |
| **GRU 4L** | **Piernas** | **88.49%** | **±7.51%** | **NB04** |

★ Mejor resultado de referencia del paper

**Nota:** Los resultados propios corresponden a entrenamiento rápido (10 épocas) para verificación. Con entrenamiento completo (500 épocas, GPU) se espera alcanzar accuracy similar al paper.

### Hallazgos Principales

1. **GRU supera a LSTM** con 25% menos parámetros
2. **Selección de articulaciones de piernas** mejora +12.93pp en GRU
3. **Alta desviación estándar** por entrenamiento limitado (10 épocas)
4. **Viabilidad clínica** demostrada con hardware de bajo costo (Kinect v2, $150 USD)

---

## Requisitos Técnicos

### Google Colab (Recomendado)
- Sin instalación necesaria
- GPU Tesla T4 gratuita
- Todos los notebooks incluyen botón "Open in Colab"

### Ejecución Local
- **Python:** 3.10, 3.11 o 3.12 (TensorFlow NO compatible con Python 3.13)
- **Dependencias:**
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
  ```

---

## Metodología

- **Framework:** TensorFlow 2.21 / Keras
- **Evaluación:** LOSO-CV (10 folds) — consistente con Jun et al. (2020)
- **Normalización:** StandardScaler ajustado dentro de cada fold (sin data leakage)
- **Métricas:** Accuracy, Sensibilidad, Especificidad, Precisión por clase

---

## Referencias

1. **Jun, K., Lee, Y., et al. (2020).** *Pathological Gait Classification Using Kinect v2 and Gated Recurrent Neural Networks.* IEEE Access, 8, 139881–139891. [DOI: 10.1109/ACCESS.2020.3013029](https://doi.org/10.1109/ACCESS.2020.3013029)

2. **Hochreiter, S., & Schmidhuber, J. (1997).** *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.

3. **Cho, K., et al. (2014).** *Learning Phrase Representations using RNN Encoder-Decoder.* EMNLP 2014.

---

## Licencia y Contacto

**Autor:** Weimar Andres Arenas Gonzalez  
**Email:** weimar.arenas@udea.edu.co  
**Institución:** Universidad de Antioquia  
**Programa:** Maestría en Ingeniería

**Repositorio:** [github.com/WeimarArenas/ProyectoDL20261](https://github.com/WeimarArenas/ProyectoDL20261)

---
