# Pipeline ICA de Grupo + Dual Regression
### Análisis de conectividad funcional en reposo (pre/post)

Este pipeline realiza un análisis de componentes independientes (ICA) a nivel de grupo sobre datos fMRI en estado de reposo, seguido de dual regression para comparar la conectividad funcional entre dos sesiones (pre y post intervención).

El flujo está dividido en dos scripts que deben correrse en orden:

```
fase1_ica_visualizacion.py  →  fase2_dual_regression.py
```

---

## Tabla de contenidos

1. [Requisitos del sistema](#1-requisitos-del-sistema)
2. [Instalación de librerías](#2-instalación-de-librerías)
3. [Estructura de datos esperada](#3-estructura-de-datos-esperada)
4. [Configuración antes de correr](#4-configuración-antes-de-correr)
5. [Fase 1 — ICA de grupo y selección de componentes](#5-fase-1--ica-de-grupo-y-selección-de-componentes)
6. [Fase 2 — Dual Regression y estadística](#6-fase-2--dual-regression-y-estadística)
7. [Archivos que genera el pipeline](#7-archivos-que-genera-el-pipeline)
8. [Parámetros ajustables](#8-parámetros-ajustables)
9. [Errores frecuentes y soluciones](#9-errores-frecuentes-y-soluciones)
10. [Referencia metodológica](#10-referencia-metodológica)

---

## 1. Requisitos del sistema

- Python 3.8 o superior
- Sistema operativo: Windows, macOS o Linux
- RAM recomendada: 16 GB o más (el ICA de grupo carga todos los sujetos en memoria)
- Espacio en disco: ~2–5 GB por corrida (mapas NIfTI por sujeto)
---

## 2. Instalación de librerías

Se recomienda crear un entorno virtual antes de instalar:

Instalar todas las dependencias:

```bash
pip install nilearn nibabel numpy scipy pandas statsmodels matplotlib
```

### Descripción de cada librería

| Librería | Versión mínima sugerida | Para qué se usa |
|---|---|---|
| `nilearn` | 0.10 | ICA de grupo (CanICA/DictLearning), maskers NIfTI, plotting neuroimagen |
| `nibabel` | 4.0 | Cargar y guardar archivos `.nii.gz` |
| `numpy` | 1.24 | Álgebra lineal, mínimos cuadrados (lstsq), manejo de arrays |
| `scipy` | 1.10 | T-test pareado, distribución t, etiquetado de clústeres 3D |
| `pandas` | 2.0 | Tabla resumen estadístico en CSV |
| `statsmodels` | 0.14 | Corrección FDR (Benjamini-Hochberg) |
| `matplotlib` | 3.7 | Visualizaciones, botones interactivos en Fase 1 |

> **Nota:** `nilearn` descarga automáticamente la máscara de materia gris ICBM152 la primera vez que se usa (requiere conexión a internet en la primera ejecución de Fase 2).

---

## 3. Estructura de datos esperada

El pipeline espera archivos BOLD ya denoisados en formato BIDS, organizados así:

```
conn/
├── sub-00/
│   ├── sub-00_ses-pre_task-rest_desc-denoised_bold.nii.gz
│   └── sub-00_ses-post_task-rest_desc-denoised_bold.nii.gz
├── sub-01/
│   ├── sub-01_ses-pre_task-rest_desc-denoised_bold.nii.gz
│   └── sub-01_ses-post_task-rest_desc-denoised_bold.nii.gz
└── ...
```

### Requisitos de los archivos BOLD

- **Espacio:** todos los sujetos deben estar normalizados al mismo espacio MNI (misma matriz voxélica, mismo shape 3D). Sujetos con shapes distintas son excluidos automáticamente.
- **Volúmenes mínimos:** cada sesión debe tener al menos 163 volúmenes temporales. Sesiones con menos volúmenes son excluidas para evitar sistemas sobredeterminados en la regresión.
- **Preprocesamiento previo asumido:** los archivos deben haber pasado ya por corrección de movimiento, normalización espacial, smoothing y denoising (regresión de señal de ruido, filtrado, etc.). Este pipeline no realiza preprocesamiento.

---

## 4. Configuración antes de correr

Al inicio de cada script hay una sección `CONFIGURACIÓN` donde debes ajustar las rutas y parámetros a tu proyecto:

```python
CONN_DIR    = Path(r"C:\tu\ruta\conn")           # carpeta con los datos BOLD
OUTPUT_DIR  = Path(r"C:\tu\ruta\resultados_ica") # donde se guardarán todos los resultados


TASK        = "rest"    # etiqueta de la tarea (parte del nombre de archivo)
TR          = 3.0       # tiempo de repetición en segundos
N_COMPONENTS = 30       # número de componentes ICA a extraer (Fase 1)
ICA_METHOD  = "canica"  # "canica" (recomendado) o "dictlearning"
```

> **Importante:** las rutas en `CONN_DIR` y `OUTPUT_DIR` deben ser **idénticas** en ambos scripts para que Fase 2 encuentre los archivos generados por Fase 1.

---

## 5. Fase 1 — ICA de grupo y selección de componentes

```bash
python fase1_ica_visualizacion.py
```

### ¿Qué hace?

#### Paso 1 — Verificación de archivos (`check_files`)
Antes de correr nada, el script recorre todos los sujetos en `SUBJECT_IDS` y verifica que:
- Existan los archivos BOLD de `ses-pre` y `ses-post`.
- Todos los archivos tengan el mismo shape 3D (misma resolución y espacio MNI).
- Cada sesión tenga al menos 163 volúmenes --> (Para este estudio, este valor se adapata según el total de volúmenes de las imágenes con las que se trabaje)

Los sujetos que no cumplan alguna condición son **excluidos automáticamente** con un mensaje explicativo en consola.

#### Paso 2 — Máscara cerebral común (`build_common_mask`)
Construye una máscara binaria que representa los vóxeles con señal BOLD válida en **todos** los sujetos y sesiones simultáneamente. Es la intersección de las máscaras individuales. Se guarda en disco para reutilizarse en Fase 2.

#### Paso 3 — ICA de grupo (`run_group_ica`)
Corre un ICA de grupo sobre todos los archivos BOLD (sujetos × sesiones) usando `CanICA` de nilearn. Si el archivo ICA ya existe en disco, lo carga directamente sin recalcular (útil si se reinicia el análisis).

Parámetros aplicados internamente:
- Suavizado espacial de 6 mm FWHM para mejorar la superposición entre sujetos.
- Detrendado y estandarización de las series temporales.
- Semilla aleatoria fija (`random_state=42`) para reproducibilidad.

El resultado es un archivo NIfTI con N mapas espaciales de componentes (uno por volumen del 4D).

#### Paso 4 — Figura resumen (`plot_all_components`)
Genera y guarda un PNG con todos los componentes en vista glass brain axial. Útil para tener una vista rápida de todos antes de la revisión detallada.

#### Paso 5 — Revisor interactivo (`interactive_component_review`)
Luego de correr el pipeline de la fase 1 se bre una ventana gráfica con controles para revisar cada componente uno por uno:

```
[← Anterior]  [Siguiente →]   navegar entre componentes
[✓ RSN]                        marcar como red neuronal válida
[✗ Ruido]                      marcar como artefacto
[Guardar y salir]              terminar y guardar selección
```

Al marcar un componente, el visualizador avanza automáticamente al siguiente. En la barra inferior se muestra cuántos componentes han sido clasificados y cuántos están marcados como RSN.

> **¿Cómo distinguir un RSN de ruido?** Los RSNs tienen distribución espacial coherente con redes cerebrales conocidas (default mode, sensoriomotora, visual, etc.). El ruido suele mostrar patrones en bordes del cerebro, senos venosos, franjas lineales, o actividad difusa y simétrica sin estructura anatómica.

#### Paso 6 — Guardar selección (`save_selection`)
Al cerrar el visualizador, la selección se guarda en:
- `componentes_seleccionados.json` — contiene los índices de los RSNs seleccionados, rutas de archivos ICA y máscara, y parámetros del análisis. Este JSON es la entrada de Fase 2.
- `RSNs_seleccionados.png` — figura con solo los componentes que marcaste como RSN.

### Archivos de salida de Fase 1

```
resultados_ica/
├── group_ICA_components_<cohort_tag>.nii.gz   # todos los mapas ICA (4D)
├── mascara_comun_<cohort_tag>.nii.gz           # máscara cerebral común
├── componentes_seleccionados.json              # selección de RSNs (entrada para Fase 2)
├── todos_los_componentes.png                   # figura resumen de todos los componentes
└── RSNs_seleccionados.png                      # figura solo con los RSNs elegidos
```

---

## 6. Fase 2 — Dual Regression y estadística

```bash
python fase2_dual_regression.py
```

### ¿Qué hace?

#### Paso 1 — Carga de selección
Lee el archivo `componentes_seleccionados.json` generado en Fase 1. Si el archivo no existe, el script termina con un error indicando que primero hay que correr Fase 1.

#### Paso 2 — Extracción de mapas RSN
Del volumen 4D del ICA de grupo, extrae únicamente los componentes marcados como RSN y los guarda como un nuevo NIfTI (`RSN_components_selected.nii.gz`).

#### Paso 3 — Z-score de los mapas ICA
CanICA guarda los componentes en una escala pequeña (~1e-2). Para que la regresión funcione correctamente, cada componente se estandariza espacialmente (media=0, std=1), llevando los valores a un rango de aproximadamente [-4, 4], compatible con las series BOLD estandarizadas.

#### Paso 4 — Dual Regression Stage 1 (regresión espacial)
Para cada sujeto y cada sesión, se ajustan los mapas ICA de grupo como regresores espaciales sobre el volumen BOLD del sujeto. El resultado son **series temporales individuales** por cada RSN — cuánto se parece la actividad de cada vóxel en cada momento al patrón espacial del RSN de grupo.

```
Entrada : mapas ICA (n_RSN × vóxeles)  +  BOLD sujeto (tiempo × vóxeles)
Salida  : timeseries (tiempo × n_RSN)
```

#### Paso 5 — Dual Regression Stage 2 (regresión temporal)
Con las series temporales individuales obtenidas en Stage 1, se hace la regresión inversa: se ajustan esas series sobre el BOLD del sujeto para obtener **mapas espaciales individuales** de cada RSN. Estos mapas reflejan la conectividad funcional específica de ese sujeto con ese RSN.

```
Entrada : timeseries (tiempo × n_RSN)  +  BOLD sujeto (tiempo × vóxeles)
Salida  : mapas individuales (n_RSN × vóxeles)
```

Los mapas se guardan por sujeto y sesión en:
```
resultados_ica/sub-XX/ses-pre/dr_stage2_RSN_maps.nii.gz
resultados_ica/sub-XX/ses-post/dr_stage2_RSN_maps.nii.gz
```

#### Paso 6 — Diagnóstico de mapas
Imprime el rango (mín/máx) y cantidad de NaNs de cada mapa individual. Sirve para verificar que el z-score funcionó: el rango esperado es ~[-5, 5]. Si el rango fuera ~[-0.05, 0.05], habría un problema de escala.

#### Paso 7 — Estadística pareada (post − pre)

Para cada RSN, se calcula la diferencia de mapas `post - pre` en cada vóxel y cada sujeto, y se aplica un **t-test pareado de una muestra** (H₀: media de la diferencia = 0). Se generan tres niveles de corrección estadística:

**FDR (Benjamini-Hochberg) con Small Volume Correction (SVC-GM)**
El FDR se aplica solo sobre vóxeles de materia gris (máscara ICBM152), reduciendo el número de comparaciones de ~187k a ~50k vóxeles y bajando el umbral de supervivencia. Con n=20 sujetos el umbral sigue siendo alto (~t>6.7), por lo que puede no haber vóxeles significativos.

**Cluster-extent correction (p<0.001 + k≥10 vóxeles)**
Estándar en neuroimagen cuando n es pequeño. Se forman clústeres con vóxeles que superen p<0.001 sin corrección, y se retienen solo los clústeres de tamaño ≥ 10 vóxeles contiguos (conectividad 26-vecinos).

**p<0.001 sin corrección**
Se incluye como referencia exploratoria. No tiene control del error tipo I, pero es útil para identificar regiones con señal antes de aplicar correcciones más estrictas.

#### Paso 8 — Visualización de resultados

Se generan dos figuras:

**Glass brain (todos los RSNs):** una fila por componente en vista glass brain cuadridreccional (lyrz), con umbral p<0.001, rojo = post>pre, azul = pre>post.

**Cortes axiales (top 3):** para los 3 componentes con más vóxeles significativos, se genera una figura con 8 cortes axiales en el plano z.

---

## 7. Archivos que genera el pipeline

### Fase 1

| Archivo | Descripción |
|---|---|
| `group_ICA_components_<tag>.nii.gz` | Todos los mapas ICA del grupo (4D NIfTI) |
| `mascara_comun_<tag>.nii.gz` | Máscara cerebral común a todos los sujetos |
| `componentes_seleccionados.json` | Índices RSN seleccionados + rutas (entrada para Fase 2) |
| `todos_los_componentes.png` | Vista rápida de todos los componentes |
| `RSNs_seleccionados.png` | Solo los componentes marcados como RSN |

### Fase 2

| Archivo | Descripción |
|---|---|
| `RSN_components_selected.nii.gz` | Mapas de los RSNs seleccionados (subconjunto del ICA) |
| `sub-XX/ses-pre/dr_stage2_RSN_maps.nii.gz` | Mapas individuales por sujeto/sesión |
| `stats/comp{XX}_tmap.nii.gz` | Mapa t completo por componente |
| `stats/comp{XX}_pmap.nii.gz` | Mapa p por componente |
| `stats/comp{XX}_thresh_FDR.nii.gz` | Mapa t umbralado con FDR (todos los vóxeles) |
| `stats/comp{XX}_thresh_FDR_post_gt_pre.nii.gz` | Solo vóxeles positivos (post>pre) FDR |
| `stats/comp{XX}_thresh_FDR_pre_gt_post.nii.gz` | Solo vóxeles negativos (pre>post) FDR |
| `stats/comp{XX}_thresh_clust_all.nii.gz` | Cluster-extent correction (ambas direcciones) |
| `stats/comp{XX}_thresh_clust_post_gt_pre.nii.gz` | Cluster positivo |
| `stats/comp{XX}_thresh_clust_pre_gt_post.nii.gz` | Cluster negativo |
| `stats/comp{XX}_thresh_p001_post_gt_pre.nii.gz` | p<0.001 sin corrección positivo |
| `stats/comp{XX}_thresh_p001_pre_gt_post.nii.gz` | p<0.001 sin corrección negativo |
| `stats/resumen_estadistico.csv` | Tabla con métricas de todos los RSNs |
| `resultados_v2_glass_p001.png` | Glass brain de todos los RSNs (p<0.001) |
| `resultados_v2_axial_top3.png` | Cortes axiales de los 3 componentes más fuertes |

---

## 8. Parámetros ajustables

### Fase 1

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `N_COMPONENTS` | 30 | Número de componentes ICA. Valores típicos: 20–70. Más componentes = mayor separación pero más ruido. |
| `ICA_METHOD` | `"canica"` | Algoritmo de descomposición. `"canica"` es más robusto; `"dictlearning"` tiende a producir componentes más localizados. |
| `TR` | 3.0 | Tiempo de repetición de tu secuencia fMRI, en segundos. |

### Fase 2

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `FDR_ALPHA` | 0.10 | Nivel α para la corrección FDR. Con n=20 se usa 0.10 (exploración permisiva); para publicación considerar 0.05. |
| `CLUSTER_FORMING_P` | 0.001 | Umbral de formación de clúster (p uncorrected). Convencional en neuroimagen. |
| `CLUSTER_K_MIN` | 10 | Tamaño mínimo de clúster en vóxeles. Aumentar para ser más conservador. |

---

## 9. Errores frecuentes y soluciones

**`No se encontró componentes_seleccionados.json`**
Hay que correr primero `fase1_ica_visualizacion.py` completo hasta que se guarde el JSON.

**`No hay sujetos válidos con ses-pre y ses-post en conn/`**
Verificar que las rutas en `CONN_DIR` sean correctas y que los nombres de archivo sigan el patrón `{sub}_{ses}_task-{TASK}_desc-denoised_bold.nii.gz`.

**`Shape (X, Y, Z) != referencia (97, 115, 97)`**
Algún sujeto tiene una normalización diferente. Ese sujeto es excluido automáticamente. Si se quiere incluirlo hay que renormalizarlo al mismo espacio MNI.

**La ventana interactiva no abre (Fase 1)**
El backend TkAgg requiere que esté instalado `tkinter`. En Linux: `sudo apt-get install python3-tk`. En servidores sin entorno gráfico se puede editar el JSON manualmente.

**Rango de mapas ~[-0.05, 0.05] en el diagnóstico de Fase 2**
Indica que el z-score no se aplicó correctamente. Verificar que el archivo ICA en el JSON corresponda al correcto y que no haya sido reemplazado por una corrida diferente.

**FDR da 0 vóxeles significativos**
Con n=10 y corrección whole-brain el umbral t requerido es muy alto (~7.4). Esto es esperable. Los resultados de cluster-extent y p<0.001 sin corrección son el nivel de análisis apropiado para este tamaño muestral (ver Eklund et al., 2016).

---

## 10. Referencia metodológica

El pipeline implementa el flujo estándar de dual regression descrito en:

- **Beckmann et al. (2009)** — Group comparison of resting-state FMRI data using multi-subject ICA and dual regression. *NeuroImage*, 47(Suppl 1), S148.
- **Filippini et al. (2009)** — Distinct patterns of brain activity in young carriers of the APOE ε4 allele. *PNAS*, 106(17), 7209–7214.

Para la corrección estadística con muestras pequeñas:

- **Eklund et al. (2016)** — Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. *PNAS*, 113(28), 7900–7905.
- **Friston et al. (1994)** — Assessing the significance of focal activations using their spatial extent. *Human Brain Mapping*, 1(3), 210–220.

La Small Volume Correction con máscara de materia gris reduce el número de comparaciones en el FDR siguiendo el principio de corrección por volumen de interés (Worsley et al., 1996).