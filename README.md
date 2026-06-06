# Clasificación de Marcha Patológica mediante Redes Recurrentes y Self-Attention

**Autor:** Weimar Andres Arenas Gonzalez  
**Curso:** Fundamentos de Deep Learning — Universidad de Antioquia  
**Dataset:** GIST Pathological Gait Dataset (Jun et al., 2020)

---

## Video de Presentación

> **[ENLACE AL VIDEO EN YOUTUBE — PENDIENTE DE SUBIR]**
> 
> *El video será subido al finalizar el proyecto y el enlace se actualizará aquí.*

---

## Descripción del Proyecto

Clasificación multiclase de **6 tipos de marcha** (1 normal + 5 patológicas) a partir de secuencias temporales de coordenadas 3D de 25 articulaciones capturadas con Kinect v2.

| Clase | Tipo de Marcha | Causa |
|-------|---------------|-------|
| 0 | Normal | — |
| 1 | Antálgica | Dolor en pie, tobillo, rodilla o cadera |
| 2 | Rígida (Stiff-legged) | Patologías articulares (artritis reumatoide) |
| 3 | Oscilante (Lurching) | Debilidad del músculo glúteo mayor |
| 4 | Steppage (Pie caído) | Debilidad del músculo tibial anterior |
| 5 | Trendelenburg | Debilidad del glúteo medio y menor |

**Problema ML:** Clasificación sequence-to-one. Input: (50 frames × 75 features). Output: clase (0-5).

---

## Estructura del Repositorio

```
ProyectoDL20261/
├── 01 - Exploracion_de_datos.ipynb        # EDA: visualización, distribución, esqueleto 3D
├── 02 - Preprocesado.ipynb                # Normalización, selección joints, LOSO-CV splits
├── 03 - Arquitectura_linea_base_RNN.ipynb # SimpleRNN 4L — referencia: 86.23%
├── 04 - LSTM_GRU.ipynb                    # LSTM/GRU 4L — referencia: 90.13%/93.67%
├── 05 - BiLSTM_BiGRU.ipynb               # Bidireccional — referencia: 90.75%
├── 06 - Transformer_SelfAttention.ipynb   # Transformer + mapas de atención
├── 07 - Comparacion_resultados.ipynb      # Comparación, métricas clínicas, resumen
├── ENTREGA1.PDF                           # Informe de la Entrega 1
├── INFORME_PROYECTO.PDF                   # Informe final (Entrega 2)
├── Patological Gait Classification.pdf   # Paper de referencia (Jun et al., 2020)
└── README.md
```

---

## Dataset

**GIST Pathological Gait Dataset** — Jun et al. (2020)

- **Fuente:** [github.com/kooksung/pathological_gait_datasets](https://github.com/kooksung/pathological_gait_datasets)
- **Formato:** CSV — `time, joint0_x, joint0_y, joint0_z, ..., joint24_x, joint24_y, joint24_z`
- **Tamaño:** 7,157 secuencias × 50 frames × 75 features
- **Clases:** Balanceadas (≈1,200 secuencias por clase)
- **Sujetos:** 10 personas

### Cómo obtener los datos

**Opción A — Google Colab (recomendada):**
```bash
git clone https://github.com/kooksung/pathological_gait_datasets.git
```

**Opción B — Google Drive:**
1. Descargar el repositorio del dataset
2. Subir la carpeta a Google Drive
3. Montar Drive en Colab y ajustar `DATA_DIR`

---

## Ejecución en Google Colab

Cada notebook tiene un botón [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)].

**Orden de ejecución:**
1. `01 - Exploracion_de_datos.ipynb` → genera `X_raw.npy`, `y_raw.npy`, `S_subjects.npy`
2. `02 - Preprocesado.ipynb` → genera `X_all.npy`, `X_legs.npy`, `metadata.pkl`
3. `03` a `06` → generan `results_rnn.pkl`, `results_lstm_gru.pkl`, `results_bidir.pkl`, `results_transformer.pkl`
4. `07 - Comparacion_resultados.ipynb` → análisis final y tabla comparativa

> **Nota:** Cada notebook de modelo (03-06) es autónomo y recarga el dataset si los `.npy` no existen.

---

## Metodología

- **Framework:** TensorFlow/Keras
- **Evaluación:** Leave-One-Subject-Out CV (LOSO-CV, 10 folds) — consistente con Jun et al. (2020)
- **Normalización:** StandardScaler ajustado *dentro de cada fold* (no data leakage)
- **Métricas:** Accuracy, Sensibilidad, Especificidad y Precisión por clase

---

## Resultados Principales

| Modelo | Accuracy LOSO-CV | Joints | Fuente |
|--------|-----------------|--------|--------|
| RNN básica | 86.23% | Todos | Jun 2020 |
| LSTM 4 capas | 87.25% | Todos | Jun 2020 |
| GRU 4 capas | 90.13% | Todos | Jun 2020 |
| **GRU 4 capas ★** | **93.67%** | **Piernas** | **Jun 2020** |
| Bi-LSTM | 90.75% | Todos | Guo 2020 |
| SimpleRNN (propio) | *ver NB 03* | Todos/Piernas | Este trabajo |
| LSTM/GRU (propio) | *ver NB 04* | Todos/Piernas | Este trabajo |
| BiLSTM/BiGRU (propio) | *ver NB 05* | Todos/Piernas | Este trabajo |
| Transformer (propio) | *ver NB 06* | Todos/Piernas | Este trabajo |

★ Mejor resultado de referencia del paper.

---

## Referencias

1. Jun, K., Lee, Y., et al. (2020). *Pathological Gait Classification Using Kinect v2 and Gated Recurrent Neural Networks.* IEEE Access, 8, 139881–139891. DOI: [10.1109/ACCESS.2020.3013029](https://doi.org/10.1109/ACCESS.2020.3013029)
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Guo, et al. (2020). Bi-LSTM for pathological gait classification.
4. Kim, J., et al. (2022). *Pathological-Gait Recognition Using Spatiotemporal Graph Convolutional Networks.* Sensors, 22(13), 4863.
