# Proyecto Neuroimagen MRI 20261

Pipeline de análisis de conectividad funcional motora en pacientes post-ACV, utilizando datos de fMRI pre y post rehabilitación.

---

## Dataset

| Campo | Detalle |
|-------|---------|
| **Nombre** | Pre-Post rehabilitation fMRI data of post-stroke patients |
| **DOI** | [10.18112/openneuro.ds003999.v1.0.2](https://doi.org/10.18112/openneuro.ds003999.v1.0.2) |
| **Formato** | BIDS 1.2.1 |
| **Licencia** | CC0 |
| **Sujetos originales** | 29 pacientes con ACV hemisférico izquierdo |
| **Sujetos procesados** | 23 (6 excluidos por falta de datos o problemas de calidad) |
| **Sesiones** | `ses-pre` (antes de rehabilitación) y `ses-post` (después) |
| **Tarea** | Reposo (resting-state) |
| **TR** | 3.0 s |

### Autores del dataset original
- Daminov V. (MD, PhD), Novak E. (MD, MSc), Slepnyova N. (MD), Mikhailov D. (MSc), Karpulevich E. (MSc)
- National Medical and Surgical Centre n.a. N.I. Pirogov, Moscow, Russia

---

## Estructura del Proyecto

```
OpenNeuro/
├── datos_originales/        # Dataset BIDS original (29 sujetos)
│   ├── participants.tsv     # Datos demográficos y clínicos
│   ├── dataset_description.json
│   └── sub-XX/              # Datos crudos por sujeto (anat/ + func/)
│
├── preprocesamiento/        # Salidas de fMRIPrep (23 sujetos)
│   ├── sub-XX/              # Datos preprocesados por sujeto
│   │   └── ses-{pre,post}/func/
│   │       ├── *_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
│   │       ├── *_desc-confounds_timeseries.tsv
│   │       └── *_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
│   ├── sub-XX.html          # Reportes QC de fMRIPrep
│   └── lesion_check/        # Verificación de lesiones en MNI
│
├── conn/                    # Salidas del pipeline de conectividad motora
│   └── sub-XX/
│       ├── *_desc-smooth_bold.nii.gz         # Imágenes suavizadas
│       ├── *_desc-denoised_bold.nii.gz       # Imágenes denoised
│       └── *_timeseries_motor_AAL.csv        # Series temporales (26 ROIs)
│
├── Grafos/                  # Análisis de teoría de grafos 
│   ├── grafos_clase.ipynb   # Notebook con análisis de grafos, métricas y dashboard 2D/3D
│   ├── demograficos.csv     # Metadatos del grupo (pre/post) de cada sujeto
│   ├── umbrales_minimos.csv # Densidad mínima requerida para mantener la red conectada por sujeto
│   ├── resultados_auc_todas_las_redes.csv  # Valores consolidados de AUC para todas las redes
│   ├── resultados_metricas_por_umbral.csv # Métricas calculadas para todos los umbrales (0.2 a 0.5)
│   └── boldconn_extracted/  # Señales BOLD piloto para pruebas portátiles rápidas
│       ├── sub-00_ses-{pre,post}_task-rest_desc-denoised_bold.nii.gz
│       └── sub-01_ses-{pre,post}_task-rest_desc-denoised_bold.nii.gz
│
├── run_fmriprep.sh          # Script de preprocesamiento (Docker + fMRIPrep)
├── run_conn_all.py          # Pipeline principal: smoothing → denoising → ROIs
├── fix_missing_subjects.py  # Normalización T1w→MNI para sub-15/sub-18 (dipy)
├── check_lesions_mni.py     # Verificación de lesiones en espacio MNI
├── run_nilearn_sub00.py     # Script piloto para sub-00
├── run_conn_sub00.m         # Script MATLAB de referencia
├── requirements.txt         # Dependencias Python
├── license.txt              # Licencia FreeSurfer
└── work/                    # Archivos intermedios de fMRIPrep (temporal)

```

---

## Guía de Scripts (para explicar el código)

Esta sección resume cada script y archivo clave en formato corto para exposición:
qué hace, qué necesita y qué produce.

### 1) run_fmriprep.sh

- **Objetivo:** Ejecutar el preprocesamiento fMRIPrep para todos los sujetos detectados en el dataset BIDS.
- **Entradas:**
  - Carpeta BIDS en `datos_originales/`
  - Licencia de FreeSurfer en `license.txt`
- **Proceso principal:**
  - Valida estructura BIDS y licencia.
  - Detecta automáticamente carpetas `sub-*`.
  - Lanza `docker run` por sujeto con fMRIPrep 24.1.1.
  - Configura salidas en MNI (`MNI152NLin2009cAsym:res-2`) y espacio anatómico.
- **Salidas:**
  - Resultados preprocesados en `preprocesamiento/`
  - Archivos intermedios en `work/`

### 2) run_conn_all.py

- **Objetivo:** Pipeline principal post-fMRIPrep para conectividad motora en los 23 sujetos.
- **Entradas:**
  - BOLD preprocesado en MNI (`preprocesamiento/sub-XX/ses-*/func/*desc-preproc_bold.nii.gz`)
  - Confounds (`*desc-confounds_timeseries.tsv`)
  - Máscaras cerebrales (`*desc-brain_mask.nii.gz`)
- **Proceso principal:**
  - Smoothing espacial (FWHM = 6 mm).
  - Denoising (26 regresores: movimiento + CSF + WM) y filtro bandpass (0.008-0.09 Hz).
  - Extracción de series temporales de 26 ROIs motoras usando atlas AAL.
  - Guarda resumen grupal y evita reprocesar sujetos completos.
- **Salidas:**
  - `conn/sub-XX/*desc-smooth_bold.nii.gz`
  - `conn/sub-XX/*desc-denoised_bold.nii.gz`
  - `conn/sub-XX/*timeseries_motor_AAL.csv`
  - `conn/group_motor_summary.csv`

### 3) fix_missing_subjects.py

- **Objetivo:** Corregir sujetos donde fMRIPrep no dejó BOLD en MNI (sub-15 y sub-18).
- **Entradas:**
  - T1w y BOLD en espacio T1w dentro de `preprocesamiento/`
- **Proceso principal:**
  - Registro T1w -> MNI con DIPY (afín: traslación/rigido/afín + SyN difeomórfico).
  - Aplica la transformación a todos los volúmenes BOLD y a la máscara cerebral.
  - Exporta con nombres compatibles con fMRIPrep para integrarse con el pipeline principal.
- **Salidas:**
  - Archivos `*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`
  - Archivos `*space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz`

### 4) check_lesions_mni.py

- **Objetivo:** Verificar que la lesión se preserve al pasar de espacio nativo a MNI.
- **Entradas:**
  - T1w nativo y T1w MNI por sujeto
  - Segmentación y máscara en MNI
- **Proceso principal:**
  - Genera figuras comparativas nativo vs MNI (cortes x/y/z).
  - Superpone segmentación sobre T1w en MNI.
  - Calcula una métrica simple de asimetría hemisférica para apoyo QC.
- **Salidas:**
  - Imágenes PNG en `preprocesamiento/lesion_check/`

### 5) run_nilearn_sub00.py

- **Objetivo:** Script piloto de validación para un solo sujeto (sub-00).
- **Entradas:**
  - Salidas fMRIPrep de sub-00 en `preprocesamiento/sub-00/`
- **Proceso principal:**
  - Smoothing, denoising y extracción de series AAL completas.
  - Calcula matrices de conectividad (pre, post y diferencia).
  - Genera figuras de conectoma y mapas de conectividad.
- **Salidas:**
  - Archivos de prueba en `conn/sub-00/` (NIfTI, CSV y PNG)

### 6) run_conn_sub00.m

- **Objetivo:** Referencia en MATLAB/CONN para sub-00 (comparación metodológica).
- **Entradas:**
  - BOLD y anatómico preprocesados por fMRIPrep
  - Máscaras de tejidos (GM, WM, CSF)
  - Confounds TSV
- **Proceso principal:**
  - Configura batch de CONN, importa dos sesiones (pre/post), aplica smoothing y denoising.
  - Define análisis seed-to-voxel de ejemplo.
- **Salidas:**
  - Proyecto y resultados de CONN en `conn/`

### 7) requirements.txt

- **Objetivo:** Fijar dependencias Python del pipeline.
- **Contenido clave:** Nilearn, Nibabel, DIPY, NumPy, SciPy, Pandas, scikit-learn y Matplotlib.
- **Uso:** `pip install -r requirements.txt`

### 8) license.txt

- **Objetivo:** Archivo de licencia de FreeSurfer requerido por fMRIPrep.
- **Nota:** Aunque se use `--fs-no-reconall`, fMRIPrep valida su presencia al iniciar.

### 9) work/

- **Objetivo:** Almacenar archivos temporales/intermedios del flujo fMRIPrep.
- **Importante:** No es salida final de análisis; puede ocupar mucho espacio.

---

## Resumen para Diapositiva (versión breve)

- `run_fmriprep.sh`: Preprocesa todos los sujetos con fMRIPrep (Docker) y genera salidas en MNI + reportes QC.
- `run_conn_all.py`: Pipeline principal en Python para smoothing, denoising y extracción de 26 ROIs motoras.
- `fix_missing_subjects.py`: Corrige casos donde faltó normalización a MNI (sub-15 y sub-18) con registro DIPY.
- `check_lesions_mni.py`: Control de calidad de lesiones, comparando nativo vs MNI con figuras y asimetría hemisférica.
- `run_nilearn_sub00.py`: Prueba completa en sub-00 para validar flujo, conectividad y visualizaciones.
- `run_conn_sub00.m`: Referencia equivalente en MATLAB/CONN para contrastar metodología con Python.
- `requirements.txt`: Lista de paquetes Python necesarios para ejecutar el pipeline.
- `license.txt`: Licencia de FreeSurfer requerida por fMRIPrep al inicio.
- `work/`: Carpeta temporal de fMRIPrep (intermedios de procesamiento, no resultados finales).

---

## Guion de Exposición en 3 Diapositivas

### Diapositiva 1 - Entrada (datos y configuración)

- Se parte del dataset BIDS crudo en `datos_originales/` con sesiones `ses-pre` y `ses-post`.
- `run_fmriprep.sh` ejecuta fMRIPrep 24.1.1 en Docker con salida en MNI 2 mm y espacio anatómico.
- Requisitos operativos: `license.txt` (FreeSurfer), dependencias de `requirements.txt` y carpeta temporal `work/`.

### Diapositiva 2 - Proceso (pipeline aplicado)

- Preprocesamiento estándar con fMRIPrep: correcciones anatómicas/funcionales, co-registro, normalización y confounds.
- Pipeline principal en `run_conn_all.py`: smoothing (6 mm), denoising (26 regresores + bandpass 0.008-0.09 Hz) y extracción de 26 ROIs motoras (AAL).
- Control de casos especiales y QC:
  - `fix_missing_subjects.py` corrige sub-15/sub-18 cuando falla la normalización a MNI.
  - `check_lesions_mni.py` verifica preservación de lesión en MNI.
  - `run_nilearn_sub00.py` y `run_conn_sub00.m` sirven como validación piloto/referencia metodológica.

### Diapositiva 3 - Salida (productos finales)

- En `preprocesamiento/`: BOLD preprocesado, máscaras, confounds y reportes HTML por sujeto.
- En `conn/sub-XX/`: imágenes `desc-smooth`, `desc-denoised` y series temporales motoras `timeseries_motor_AAL.csv`.
- En `conn/`: resumen grupal (`group_motor_summary.csv`) para análisis posterior de conectividad y estadística.

---

## Pipeline de Procesamiento

### Paso 1: Preprocesamiento con fMRIPrep

Se utilizó **fMRIPrep 24.1.1** (vía Docker) para el preprocesamiento estándar de las imágenes fMRI.

**Script:** `run_fmriprep.sh`

**Configuración:**
- Espacio de salida: `MNI152NLin2009cAsym:res-2` + `anat`
- FreeSurfer: deshabilitado (`FS_FLAG=0`)
- CPUs: 8 | Memoria: 16 GB

**Resultado:** Imágenes BOLD preprocesadas en espacio MNI, máscaras cerebrales y matrices de confounds para los 23 sujetos.

**Casos especiales:**
| Sujeto | Problema | Solución |
|--------|----------|----------|
| sub-15, sub-18 | fMRIPrep no normalizó a MNI (solo espacio T1w) debido a lesiones extensas | Normalización manual con **dipy** (registro afín + SyN diffeomórfico) — `fix_missing_subjects.py` |
| sub-30 | fMRIPrep nunca se ejecutó | Se ejecutó fMRIPrep individualmente vía Docker |
| sub-35 | ses-pre tiene solo 30 volúmenes (vs 163 en ses-post) | Filtro bandpass omitido para series cortas (<34 volúmenes) |

### Paso 2: Smoothing Espacial

- **Método:** Gaussian smoothing (nilearn `image.smooth_img`)
- **FWHM:** 6 mm
- **Salida:** `*_desc-smooth_bold.nii.gz`

### Paso 3: Denoising (Limpieza de señal)

- **Método:** `nilearn.image.clean_img`
- **Regresores de confounds** (26 total):
  - 6 parámetros de movimiento rígido
  - 6 derivadas temporales del movimiento
  - 6 parámetros de movimiento al cuadrado
  - 6 derivadas al cuadrado
  - CSF (señal de líquido cefalorraquídeo)
  - Materia blanca (WM)
- **Filtro bandpass:** 0.008 – 0.09 Hz
- **Estandarización:** z-score por muestra
- **Salida:** `*_desc-denoised_bold.nii.gz`

### Paso 4: Extracción de Series Temporales — Corteza Motora

- **Atlas:** AAL (SPM12) — 117 ROIs totales
- **ROIs seleccionadas:** 26 regiones de la red motora
- **Método:** `nilearn.maskers.NiftiLabelsMasker`
- **Salida:** `*_timeseries_motor_AAL.csv` (26 columnas × N volúmenes)

#### ROIs de la Red Motora (26)

| Región | Hemisferio |
|--------|-----------|
| Precentral (M1 — corteza motora primaria) | L, R |
| Supp_Motor_Area (SMA — área motora suplementaria) | L, R |
| Postcentral (S1 — corteza somatosensorial primaria) | L, R |
| Paracentral_Lobule (lobulillo paracentral) | L, R |
| Cerebelum_Crus1 | L, R |
| Cerebelum_Crus2 | L, R |
| Cerebelum_3 | L, R |
| Cerebelum_4_5 | L, R |
| Cerebelum_6 | L, R |
| Cerebelum_7b | L, R |
| Cerebelum_8 | L, R |
| Cerebelum_9 | L, R |
| Cerebelum_10 | L, R |

---

## Sujetos Procesados (23)

```
sub-00  sub-01  sub-11  sub-12  sub-13  sub-15  sub-16  sub-17  sub-18
sub-20  sub-21  sub-22  sub-23  sub-24  sub-25  sub-26  sub-27  sub-28
sub-29  sub-30  sub-33  sub-34  sub-35
```

**Sujetos excluidos** (6): sub-02, sub-03, sub-05, sub-07, sub-10, sub-14 — no cumplieron criterios de calidad o datos incompletos.

---

## Estado de los Análisis
 
### Análisis
1. **Matriz de correlación:** Extracción de series temporales del BOLD preprocesado y denoised utilizando el atlas de Schaefer (400 ROIs, 7 redes de Yeo). Cálculo de matrices de conectividad funcional de Pearson completas para todas las sesiones de los sujetos.
2. **Análisis de grafos y Visualización:** 
   - **Cálculo de Umbrales:** Búsqueda automatizada de densidad de umbral mínimo que mantiene el grafo conectado por sujeto.
   - **Métricas Complejas:** Cálculo de métricas globales (Eficiencia, Modularidad, Grado Promedio, Clustering) y nodales (Grado, Betweenness, Clustering, Eficiencia) a lo largo de un rango proporcional de umbrales (0.2 a 0.5).
   - **Integración por AUC:** Integración de métricas mediante el Área Bajo la Curva (AUC) para obtener valores robustos de análisis.
   - **Dashboard Interactivo Portátil:** Visualizador dinámico.

### Análisis Pendientes
3. **ICA** (Análisis de Componentes Independientes) para comparar espacialmente redes de estado de reposo (RSNs).

---

## Instalación y Uso

### Requisitos
- Python 3.10+
- Docker Desktop (para fMRIPrep)
- ~50 GB de espacio en disco para datos y archivos intermedios

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Ejecución

```bash
# 1. Preprocesamiento con fMRIPrep (requiere Docker)
chmod +x run_fmriprep.sh
./run_fmriprep.sh

# 2. Corrección de sujetos con normalización fallida (si aplica)
python fix_missing_subjects.py

# 3. Pipeline principal: Smoothing → Denoising → Series Temporales
python run_conn_all.py
```

---

## Herramientas y Versiones

| Herramienta | Versión | Uso |
|-------------|---------|-----|
| fMRIPrep | 24.1.1 | Preprocesamiento estándar de fMRI |
| Docker | 28.4.0 | Contenedor para fMRIPrep |
| nilearn | 0.13.1 | Smoothing, denoising, extracción de ROIs |
| nibabel | 5.4.2 | Lectura/escritura de imágenes NIfTI |
| dipy | 1.11.0 | Registro T1w→MNI (diffeomórfico) |
| bctpy | 0.5.2 | Cálculo de métricas de redes complejas (Teoría de Grafos) |
| networkx | 3.0 | Análisis de grafos, modelado de redes y comprobaciones |
| numpy | 2.2.6 | Computación numérica |
| pandas | 2.3.3 | Manejo de tablas y CSVs |
| scipy | 1.15.3 | Filtros y procesamiento de señal |
| scikit-learn | 1.7.2 | Dependencia de nilearn |
| plotly | 5.15.0 | Renderizado de gráficos interactivos 2D y Red Cerebral 3D |
| ipywidgets | 8.0.0 | Controles interactivos y lógica de filtrado del Dashboard |
| matplotlib | 3.10.8 | Visualización estática |
| Python | 3.10.11 | Lenguaje principal |

---

## Licencia

El dataset original está bajo licencia **CC0**. El código de este proyecto es de uso académico.
