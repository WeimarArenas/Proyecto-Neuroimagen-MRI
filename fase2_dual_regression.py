"""
=======================================================================
FASE 2 — DUAL REGRESSION CON LOS COMPONENTES SELECCIONADOS
=======================================================================

Corre este script DESPUÉS de fase1_ica_visualizacion.py.

Lee automáticamente desde el JSON de fase 1:
  - La ruta exacta del archivo ICA (sin importar el nombre)
  - La ruta de la máscara común
  - Los índices de los componentes RSN seleccionados

Flujo:
  1. Carga selección de componentes (JSON de fase 1)
  2. Extrae solo los mapas ICA de los RSNs elegidos
  3. Z-scorea los mapas ICA para normalizar escala
  4. Dual Regression Stage 1 → timeseries por sujeto
  5. Dual Regression Stage 2 → mapas espaciales por sujeto
  6. T-test pareado (post − pre) + corrección FDR
  7. Visualización de resultados

FIXES aplicados vs versión original:
  [FIX 1] NaNs en p_vals → 1.0 antes del FDR
  [FIX 2] t_vals/p_vals forzados a np.array 1D
  [FIX 3] sig_mask = 0 en posiciones NaN
  [FIX 4] t_clean sin NaN para inverse_transform
  [FIX 5] Diagnóstico de mapas por sujeto
  [FIX 6] Shapes documentadas en stage1/stage2
  [FIX 7] p<0.05 sin corrección impreso para confirmar señal
  [FIX 8] Masker separado sin standardize para mapas ICA
  [FIX 9] Rutas ICA/máscara leídas desde JSON → no más mismatch de nombres
  [FIX 10] Z-score manual de comp_data → escala correcta para lstsq
=======================================================================
"""

import numpy as np
import pandas as pd
import nibabel as nib
import json
import hashlib
from pathlib import Path
from scipy import stats
from scipy.ndimage import label as nd_label
from statsmodels.stats.multitest import fdrcorrection

from nilearn.maskers import NiftiMasker
from nilearn.image import index_img, concat_imgs, resample_to_img
from nilearn import plotting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# =======================================================================
# CONFIGURACIÓN  ← cambia solo estas líneas
# =======================================================================

CONN_DIR    = Path(r"C:\Users\ASUS\Desktop\OpenNeuro\conn")
OUTPUT_DIR  = Path(r"C:\Users\ASUS\Desktop\OpenNeuro\ica\resultados_ica")

# SUBJECT_IDS se detecta automáticamente desde conn/ en detect_subjects()
SUBJECT_IDS: list = []

TASK              = "rest"
TR                = 3.0
MIN_VOLUMES       = 163    # mínimo de volúmenes por sesión
FDR_ALPHA         = 0.10   # exploración permisiva (n=20 sujetos; whole-brain FDR requiere t>7.39, raramente alcanzable)
# Cluster-extent correction (estándar neuroimagen con n pequeño):
#   1) umbral de formación de clúster p<CLUSTER_FORMING_P uncorrected
#   2) se retienen clústeres de tamaño ≥ CLUSTER_K_MIN vóxeles
#   Con TR=3s y n=20 estos valores son convencionales (Eklund et al., 2016).
CLUSTER_FORMING_P = 0.001  # umbral de vóxel (uncorrected) para formar clústeres
CLUSTER_K_MIN     = 10     # tamaño mínimo de clúster en vóxeles (ajustable)

# Rutas de respaldo (se sobreescriben con las del JSON tras detect_subjects)
SELECTION_FILE = OUTPUT_DIR / "componentes_seleccionados.json"
STATS_DIR      = OUTPUT_DIR / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


# =======================================================================
# UTILIDADES
# =======================================================================

def get_bold_path(subject: str, session: str) -> Path:
    return (
        CONN_DIR / subject /
        f"{subject}_{session}_task-{TASK}_desc-denoised_bold.nii.gz"
    )

def detect_subjects() -> list:
    """
    Detecta automáticamente los sujetos válidos en CONN_DIR.
    Criterios (igual que Fase 1):
      - Carpeta sub-XX existe en conn/
      - Tiene ses-pre y ses-post con archivos denoised_bold
      - Shape espacial consistente entre sujetos
      - ≥ MIN_VOLUMES volúmenes por sesión
    """
    global SUBJECT_IDS
    print("Detectando sujetos automáticamente desde conn/...")

    all_subjects = sorted(
        d.name for d in CONN_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"  Encontrados {len(all_subjects)} sujetos potenciales: {all_subjects}")

    valid_subjects = []
    ref_shape = None

    for sub in all_subjects:
        ok = True
        for ses in ["ses-pre", "ses-post"]:
            p = get_bold_path(sub, ses)
            if not p.exists():
                print(f"  EXCLUIDO {sub}: falta {p.name}")
                ok = False
                break
            img = nib.load(str(p))
            shape3d = img.shape[:3]
            if ref_shape is None:
                ref_shape = shape3d
            elif shape3d != ref_shape:
                print(f"  EXCLUIDO {sub}: shape {shape3d} != referencia {ref_shape}")
                ok = False
                break
            n_vols = img.shape[3] if len(img.shape) > 3 else 1
            if n_vols < MIN_VOLUMES:
                print(f"  EXCLUIDO {sub}/{ses}: solo {n_vols} volúmenes (mín={MIN_VOLUMES})")
                ok = False
                break
        if ok:
            valid_subjects.append(sub)

    SUBJECT_IDS = valid_subjects
    if not SUBJECT_IDS:
        raise FileNotFoundError("No hay sujetos válidos con ses-pre y ses-post en conn/")

    print(f"  {len(SUBJECT_IDS)} sujetos válidos (shape {ref_shape}), "
          f"{len(SUBJECT_IDS)*2} imágenes BOLD.\n")
    return SUBJECT_IDS

def load_selection() -> dict:
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {SELECTION_FILE}\n"
            "Asegúrate de haber corrido primero fase1_ica_visualizacion.py"
        )
    with open(SELECTION_FILE) as f:
        sel = json.load(f)
    print("Selección cargada:")
    print(f"  RSN indices     : {sel['rsn_indices']}")
    print(f"  Total comp ICA  : {sel['n_total']}")
    print(f"  Método ICA      : {sel.get('ica_method', 'canica')}")
    print(f"  Archivo ICA     : {sel.get('ica_file', 'no registrado')}")
    print()
    return sel

def extract_rsn_components(components_img, rsn_indices):
    rsn_imgs       = [index_img(components_img, i) for i in rsn_indices]
    rsn_components = concat_imgs(rsn_imgs)
    out = OUTPUT_DIR / "RSN_components_selected.nii.gz"
    rsn_components.to_filename(str(out))
    print(f"  Mapas RSN extraídos ({len(rsn_indices)} componentes): {out.name}\n")
    return rsn_components

def zscore_components(comp_data: np.ndarray) -> np.ndarray:
    """
    [FIX 10] Z-scorea cada componente ICA por separado.

    CanICA/DictLearning de nilearn guarda los componentes con valores
    en escala ~1e-2, lo que produce coeficientes de stage1 minúsculos
    y mapas stage2 aplastados (rango ~[-0.05, 0.05]).
    Al z-scorear comp_data obtenemos rango ~[-4, 4] compatible con
    los BOLDs estandarizados, y los mapas stage2 salen en escala
    de beta-maps de conectividad (~[-5, 5]).

    comp_data shape: (n_rsn, n_voxeles)
    """
    means = comp_data.mean(axis=1, keepdims=True)
    stds  = comp_data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (comp_data - means) / stds

def build_maskers(mask_img):
    """
    [FIX 8] Dos maskers separados:
      masker_bold : standardize=True  → para los BOLDs de cada sujeto
      masker_ica  : standardize=False → para extraer comp_data sin alterar escala
                    (la normalización la hace zscore_components)
    """
    masker_bold = NiftiMasker(
        mask_img=mask_img, t_r=TR,
        high_pass=None, low_pass=None,
        detrend=True, standardize=True,
        memory="nilearn_cache", memory_level=1, verbose=0,
    )
    masker_bold.fit(str(get_bold_path(SUBJECT_IDS[0], "ses-pre")))

    masker_ica = NiftiMasker(
        mask_img=mask_img,
        detrend=False, standardize=False,
        memory="nilearn_cache", memory_level=1, verbose=0,
    )
    masker_ica.fit(str(get_bold_path(SUBJECT_IDS[0], "ses-pre")))

    return masker_bold, masker_ica


def get_gm_mask(masker_bold: NiftiMasker) -> np.ndarray:
    """
    Descarga la máscara de materia gris ICBM152 de nilearn y la proyecta
    al espacio de la máscara cerebral del masker.

    Reducir las comparaciones de 187k → ~50k vóxeles baja el umbral
    Benjamini-Hochberg de t>7.39 a t>6.72, permitiendo que componentes
    con t_max ≥ 6.97 (comp#21) sobrevivan la corrección FDR.

    Retorna: array booleano (n_vox,)  —  True = vóxel de materia gris.
    """
    from nilearn.datasets import fetch_icbm152_2009

    print("  Cargando máscara de materia gris ICBM152...")
    icbm   = fetch_icbm152_2009()
    gm_img = nib.load(icbm["gm"])

    # Remuestrear al espacio de la máscara cerebral (97×115×97, 2 mm)
    gm_res  = resample_to_img(gm_img, masker_bold.mask_img_,
                               interpolation="continuous")
    gm_data = gm_res.get_fdata() > 0.5    # umbral estricto p(GM) > 0.5 (~50k vox)

    # Índices de los vóxeles dentro de la máscara cerebral
    brain_mask_data = masker_bold.mask_img_.get_fdata().astype(bool)
    brain_coords    = np.where(brain_mask_data)
    gm_in_mask      = gm_data[brain_coords]              # (n_vox,) booleano

    n_gm    = int(gm_in_mask.sum())
    n_total = len(gm_in_mask)
    print(f"  GM vóxeles: {n_gm:,} / {n_total:,} "
          f"({100 * n_gm / n_total:.1f}% de la máscara cerebral)\n")
    return gm_in_mask


# =======================================================================
# DIAGNÓSTICO DE MAPAS INDIVIDUALES  [FIX 5]
# =======================================================================

def diagnostico_mapas(pre_maps: np.ndarray, post_maps: np.ndarray):
    """
    Imprime min/max/NaNs por sujeto.
    Si ves rango ~[-0.05, 0.05] el z-score no funcionó.
    Si ves rango ~[-5, 5] el pipeline está correcto.
    """
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE MAPAS INDIVIDUALES (stage2)")
    print("  Rango esperado: ~[-5, 5]  |  Rango problemático: ~[-0.05, 0.05]")
    print("=" * 60)
    for i, sub in enumerate(SUBJECT_IDS):
        pre  = pre_maps[i]
        post = post_maps[i]
        print(f"\n  {sub}:")
        print(f"    PRE  → min={pre.min():.4f}  max={pre.max():.4f}  "
              f"NaNs={np.isnan(pre).sum()}  zeros={(pre==0).sum()}")
        print(f"    POST → min={post.min():.4f}  max={post.max():.4f}  "
              f"NaNs={np.isnan(post).sum()}  zeros={(post==0).sum()}")
    print("=" * 60 + "\n")


# =======================================================================
# DUAL REGRESSION
# =======================================================================

def stage1(bold_path: str, comp_data: np.ndarray,
           masker: NiftiMasker) -> np.ndarray:
    """
    Regresión espacial: mapas RSN grupo → timeseries por sujeto.
    comp_data : (n_rsn, V)
    retorna   : (T, n_rsn)
    """
    datos = masker.transform(bold_path)                        # (T, V)
    ts, _, _, _ = np.linalg.lstsq(comp_data.T, datos.T, rcond=None)
    return ts.T                                                # (T, n_rsn)

def stage2(bold_path: str, timeseries: np.ndarray,
           masker: NiftiMasker) -> np.ndarray:
    """
    Regresión temporal: timeseries → mapas espaciales por sujeto.
    timeseries : (T, n_rsn)
    retorna    : (n_rsn, V)
    """
    datos = masker.transform(bold_path)                        # (T, V)
    mapas, _, _, _ = np.linalg.lstsq(timeseries, datos, rcond=None)
    return mapas                                               # (n_rsn, V)

def run_dual_regression(rsn_components, rsn_indices, masker_bold, masker_ica):
    print(f"Dual Regression — {len(rsn_indices)} RSNs × {len(SUBJECT_IDS)} sujetos...\n")

    # [FIX 8] Extraer comp_data sin standardize
    comp_data_raw = masker_ica.transform(rsn_components)       # (n_rsn, V)
    print(f"  comp_data raw   — rango: [{comp_data_raw.min():.4f}, {comp_data_raw.max():.4f}]")

    # [FIX 10] Z-score para llevar a escala compatible con BOLDs estandarizados
    comp_data = zscore_components(comp_data_raw)
    print(f"  comp_data zscore— rango: [{comp_data.min():.3f}, {comp_data.max():.3f}]")
    if comp_data.max() - comp_data.min() < 1.0:
        print("  *** ADVERTENCIA: rango todavía pequeño después del z-score")
        print("      Revisa que los mapas ICA tengan varianza espacial ***")
    print()

    pre_maps_list  = []
    post_maps_list = []

    for i, sub in enumerate(SUBJECT_IDS):
        print(f"  [{i+1}/{len(SUBJECT_IDS)}] {sub}")

        for session, maps_list in [("ses-pre",  pre_maps_list),
                                   ("ses-post", post_maps_list)]:
            bold_path = str(get_bold_path(sub, session))

            ts     = stage1(bold_path, comp_data, masker_bold)
            s_maps = stage2(bold_path, ts, masker_bold)
            maps_list.append(s_maps)

            out_dir = OUTPUT_DIR / sub / session
            out_dir.mkdir(parents=True, exist_ok=True)
            masker_bold.inverse_transform(s_maps).to_filename(
                str(out_dir / "dr_stage2_RSN_maps.nii.gz")
            )
            print(f"    {session}: ts{ts.shape} → maps{s_maps.shape}  "
                  f"| rango [{s_maps.min():.3f}, {s_maps.max():.3f}]")

    print()
    return np.array(pre_maps_list), np.array(post_maps_list)


# =======================================================================
# CLUSTER-EXTENT CORRECTION
# =======================================================================

def cluster_extent_correction(t_vals: np.ndarray, p_vals: np.ndarray,
                               masker: NiftiMasker,
                               forming_p: float = CLUSTER_FORMING_P,
                               k_min: int = CLUSTER_K_MIN,
                               direction: str = "both") -> np.ndarray:
    """
    Cluster-extent correction estándar para neuroimagen con n pequeño.

    Pasos:
      1) Umbral de formación de clúster: retener vóxeles con p < forming_p
         (uncorrected). Esto es convencional (Friston et al., 1994).
      2) Separar en conglomerados 3D contiguos (conectividad 26-vecinos).
      3) Retener solo clústeres con k ≥ k_min vóxeles.
      4) Devolver máscara booleana de vóxeles supervivientes.

    direction : "pos"  → solo t>0 (post>pre)
                "neg"  → solo t<0 (pre>post)
                "both" → ambas direcciones por separado y unidas
    """
    # Reconstruir volumen 3D a través del masker
    mask_img   = masker.mask_img_
    mask_data  = mask_img.get_fdata().astype(bool)
    vol_shape  = mask_data.shape

    df_val     = len(SUBJECT_IDS) - 1
    from scipy.stats import t as t_dist
    t_thresh   = float(t_dist.ppf(1 - forming_p / 2, df=df_val))  # bilateral → divide por 2

    survive = np.zeros(len(t_vals), dtype=bool)

    for sign, t_sign_thresh in [("pos", t_thresh), ("neg", -t_thresh)]:
        if direction == "pos" and sign == "neg":
            continue
        if direction == "neg" and sign == "pos":
            continue

        if sign == "pos":
            candidate = (t_vals > t_sign_thresh) & ~np.isnan(t_vals)
        else:
            candidate = (t_vals < t_sign_thresh) & ~np.isnan(t_vals)

        # Proyectar máscara candidata al volumen 3D
        vol = np.zeros(vol_shape, dtype=bool)
        coords = np.where(mask_data)
        vol[coords] = candidate

        # Etiquetar clústeres 3D (26-conectividad)
        struct = np.ones((3, 3, 3), dtype=int)
        labeled, n_clusters = nd_label(vol, structure=struct)

        for clust_id in range(1, n_clusters + 1):
            clust_mask_3d = labeled == clust_id
            if clust_mask_3d.sum() >= k_min:
                # Mapear de vuelta al espacio vectorial del masker
                vox_in_mask = clust_mask_3d[coords]
                survive |= vox_in_mask

    return survive


# =======================================================================
# ESTADÍSTICA PAREADA
# =======================================================================

def paired_stats(pre_maps: np.ndarray, post_maps: np.ndarray,
                 rsn_indices: list, masker_bold: NiftiMasker,
                 gm_mask: np.ndarray = None) -> pd.DataFrame:
    """
    T-test pareado (post − pre) vóxel a vóxel + corrección FDR.
    Si se provee gm_mask, el FDR se aplica solo sobre vóxeles de materia
    gris (Small Volume Correction), reduciendo m y bajando el umbral BH.
    """
    print(f"Estadística pareada — FDR α={FDR_ALPHA}...\n")

    n_suj, n_rsn, n_vox = pre_maps.shape
    diff = post_maps - pre_maps                                # (n_suj, n_rsn, V)
    print(f"  Sujetos: {n_suj}  |  RSNs: {n_rsn}  |  Vóxeles: {n_vox}\n")

    resumen = []

    for rsn_pos, comp_idx in enumerate(rsn_indices):
        comp_diff = diff[:, rsn_pos, :]                        # (n_suj, V)

        # T-test
        result = stats.ttest_1samp(comp_diff, popmean=0, axis=0)

        # [FIX 2] Forzar a np.ndarray 1D
        t_vals = np.asarray(result.statistic).ravel()
        p_vals = np.asarray(result.pvalue).ravel()

        # [FIX 7] Diagnóstico sin corrección
        n_p05  = int((p_vals < 0.05).sum())
        n_p001 = int((p_vals < 0.001).sum())
        print(f"  comp{comp_idx:02d} | max|t|={np.nanmax(np.abs(t_vals)):.2f} | "
              f"p<0.05 sin corr={n_p05} | p<0.001 sin corr={n_p001}")

        # [FIX 1] NaNs → p=1.0
        nan_mask = np.isnan(p_vals) | np.isnan(t_vals)
        p_clean  = p_vals.copy()
        p_clean[nan_mask] = 1.0

        # [FIX 4] t_clean para operaciones y guardado
        t_clean = np.nan_to_num(t_vals, nan=0.0)

        # ── FDR — Small Volume Correction (GM) ──────────────────────────
        # Con n=20 y ~50k vóxeles de GM, el umbral BH es t>6.7 aprox.
        # Raramente alcanzable en resting-state; los mapas suelen dar 0.
        reject_fdr = np.zeros(len(p_clean), dtype=bool)
        if gm_mask is not None:
            gm_valid = gm_mask & ~nan_mask
            if gm_valid.sum() > 0:
                gm_reject, _ = fdrcorrection(p_clean[gm_valid], alpha=FDR_ALPHA)
                reject_fdr[gm_valid] = gm_reject
        else:
            reject_fdr, _ = fdrcorrection(p_clean, alpha=FDR_ALPHA)
        reject_fdr[nan_mask] = False

        sig_mask_fdr = reject_fdr.astype(float)
        sig_t_pos_fdr = t_clean * sig_mask_fdr * (t_clean > 0)
        sig_t_neg_fdr = t_clean * sig_mask_fdr * (t_clean < 0)
        sig_t_all_fdr = t_clean * sig_mask_fdr

        # ── Cluster-extent correction (p<0.001 + k≥CLUSTER_K_MIN) ──────
        # Estándar para estudios con n pequeño cuando FDR whole-brain falla.
        # Referencia: Eklund et al. (2016) PNAS; Friston et al. (1994).
        clust_survive = cluster_extent_correction(
            t_vals, p_vals, masker_bold,
            forming_p=CLUSTER_FORMING_P, k_min=CLUSTER_K_MIN,
            direction="both",
        )
        clust_survive[nan_mask] = False
        sig_mask_clust = clust_survive.astype(float)
        sig_t_pos_clust = t_clean * sig_mask_clust * (t_clean > 0)
        sig_t_neg_clust = t_clean * sig_mask_clust * (t_clean < 0)
        sig_t_all_clust = t_clean * sig_mask_clust

        # ── p<0.001 sin corrección (exploración, referencia) ──────────
        p001_mask     = ((p_clean < 0.001) & ~nan_mask).astype(float)
        sig_t_pos_001 = t_clean * p001_mask * (t_clean > 0)
        sig_t_neg_001 = t_clean * p001_mask * (t_clean < 0)

        # Guardar mapas
        prefix = STATS_DIR / f"comp{comp_idx:02d}"
        masker_bold.inverse_transform(t_clean        ).to_filename(f"{prefix}_tmap.nii.gz")
        masker_bold.inverse_transform(p_clean        ).to_filename(f"{prefix}_pmap.nii.gz")
        # FDR maps
        masker_bold.inverse_transform(sig_t_all_fdr ).to_filename(f"{prefix}_thresh_FDR.nii.gz")
        masker_bold.inverse_transform(sig_t_pos_fdr ).to_filename(f"{prefix}_thresh_FDR_post_gt_pre.nii.gz")
        masker_bold.inverse_transform(sig_t_neg_fdr ).to_filename(f"{prefix}_thresh_FDR_pre_gt_post.nii.gz")
        # Cluster-extent maps
        masker_bold.inverse_transform(sig_t_all_clust).to_filename(f"{prefix}_thresh_clust_all.nii.gz")
        masker_bold.inverse_transform(sig_t_pos_clust).to_filename(f"{prefix}_thresh_clust_post_gt_pre.nii.gz")
        masker_bold.inverse_transform(sig_t_neg_clust).to_filename(f"{prefix}_thresh_clust_pre_gt_post.nii.gz")
        # p<0.001 maps
        masker_bold.inverse_transform(sig_t_pos_001).to_filename(f"{prefix}_thresh_p001_post_gt_pre.nii.gz")
        masker_bold.inverse_transform(sig_t_neg_001).to_filename(f"{prefix}_thresh_p001_pre_gt_post.nii.gz")

        n_sig_pos_fdr   = int((sig_t_pos_fdr   != 0).sum())
        n_sig_neg_fdr   = int((sig_t_neg_fdr   != 0).sum())
        n_sig_pos_clust = int((sig_t_pos_clust != 0).sum())
        n_sig_neg_clust = int((sig_t_neg_clust != 0).sum())
        n_p001_pos      = int((sig_t_pos_001 != 0).sum())
        n_p001_neg      = int((sig_t_neg_001 != 0).sum())
        max_t           = float(np.nanmax(np.abs(t_vals)))
        print(f"         → FDR(SVC-GM): post>pre={n_sig_pos_fdr}vox | pre>post={n_sig_neg_fdr}vox")
        print(f"         → Clust(p<{CLUSTER_FORMING_P},k≥{CLUSTER_K_MIN}): "
              f"post>pre={n_sig_pos_clust}vox | pre>post={n_sig_neg_clust}vox")
        print(f"         → p<.001 uncorr: post>pre={n_p001_pos}vox | pre>post={n_p001_neg}vox\n")

        resumen.append({
            "componente_ICA"          : comp_idx,
            "max_abs_t"               : round(max_t, 3),
            "p_lt_005_raw"            : n_p05,
            "p_lt_001_raw"            : n_p001,
            "vox_post_gt_pre_FDR"     : n_sig_pos_fdr,
            "vox_pre_gt_post_FDR"     : n_sig_neg_fdr,
            "total_sig_FDR"           : n_sig_pos_fdr + n_sig_neg_fdr,
            "vox_post_gt_pre_clust"   : n_sig_pos_clust,
            "vox_pre_gt_post_clust"   : n_sig_neg_clust,
            "total_sig_clust"         : n_sig_pos_clust + n_sig_neg_clust,
            "vox_post_gt_pre_p001"    : n_p001_pos,
            "vox_pre_gt_post_p001"    : n_p001_neg,
            "total_sig_p001"          : n_p001_pos + n_p001_neg,
            "pct_sig_FDR"             : round(100 * (n_sig_pos_fdr   + n_sig_neg_fdr)   / n_vox, 2),
            "pct_sig_clust"           : round(100 * (n_sig_pos_clust + n_sig_neg_clust) / n_vox, 2),
            "pct_sig_p001"            : round(100 * (n_p001_pos + n_p001_neg) / n_vox, 2),
        })

    df = pd.DataFrame(resumen)
    df.to_csv(STATS_DIR / "resumen_estadistico.csv", index=False)
    print(f"  CSV guardado: {STATS_DIR / 'resumen_estadistico.csv'}\n")
    return df

# =======================================================================
# VISUALIZACIÓN MEJORADA
# =======================================================================

def plot_results_v2(resumen: pd.DataFrame, rsn_indices: list, masker_bold):
    """
    Una figura por componente RSN con los mapas p<.001 uncorr.
    Positivos (post>pre) en rojo/amarillo, negativos (pre>post) en azul/cian.
    """
    from nilearn import plotting
    from scipy.stats import t as t_dist

    df_val  = len(SUBJECT_IDS) - 1
    t_thresh_001 = float(t_dist.ppf(1 - 0.001, df=df_val))  # one-tailed → bilateral via sign

    for _, row in resumen.iterrows():
        comp_idx = int(row["componente_ICA"])
        prefix   = str(STATS_DIR / f"comp{comp_idx:02d}")
        tmap_path = f"{prefix}_tmap.nii.gz"

        if not Path(tmap_path).exists():
            continue

        tmap = nib.load(tmap_path)
        t_data = tmap.get_fdata()
        max_t  = float(np.nanmax(np.abs(t_data)))

        fig, axes = plt.subplots(2, 1, figsize=(14, 6))

        # Post > Pre (t > 0)
        plotting.plot_stat_map(
            tmap, threshold=t_thresh_001,
            display_mode="z", cut_coords=7,
            title=f"Comp #{comp_idx} | post>pre | p<.001 | max|t|={max_t:.2f}",
            axes=axes[0], black_bg=False, colorbar=True,
        )
        # Pre > Post (t < 0) — invertir signo
        neg_img = nib.Nifti1Image(-t_data, tmap.affine, tmap.header)
        plotting.plot_stat_map(
            neg_img, threshold=t_thresh_001,
            display_mode="z", cut_coords=7,
            title=f"Comp #{comp_idx} | pre>post | p<.001",
            axes=axes[1], black_bg=False, colorbar=True,
        )

        plt.tight_layout()
        out = STATS_DIR / f"comp{comp_idx:02d}_resultados.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figura guardada: {out.name}")


# =======================================================================
# MAIN
# =======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 2 — DUAL REGRESSION + ESTADÍSTICA PAREADA")
    print("=" * 60 + "\n")

    # 0. Detectar sujetos automáticamente desde conn/
    detect_subjects()
    COHORT_TAG = (
        f"{len(SUBJECT_IDS)}subs_"
        f"{hashlib.sha1(','.join(SUBJECT_IDS).encode('utf-8')).hexdigest()[:8]}"
    )
    ICA_FILE  = OUTPUT_DIR / f"group_ICA_components_{COHORT_TAG}.nii.gz"
    MASK_FILE = OUTPUT_DIR / f"mascara_comun_{COHORT_TAG}.nii.gz"

    # 1. Cargar selección de Fase 1
    sel         = load_selection()
    rsn_indices = sel["rsn_indices"]

    # Rutas ICA y máscara desde JSON (sobreescribe las de respaldo)
    ica_path  = sel.get("ica_file",  str(ICA_FILE))
    mask_path = sel.get("mask_file", str(MASK_FILE))

    components_img = nib.load(ica_path)
    mask_img       = nib.load(mask_path)
    print(f"ICA cargado   : {Path(ica_path).name}")
    print(f"Máscara       : {Path(mask_path).name}")
    print(f"RSNs a procesar: {rsn_indices}\n")

    # 2. Extraer mapas RSN seleccionados
    rsn_components = extract_rsn_components(components_img, rsn_indices)

    # 3. Construir maskers
    masker_bold, masker_ica = build_maskers(mask_img)

    # 4. Máscara de materia gris para SVC-FDR
    gm_mask = get_gm_mask(masker_bold)

    # 5. Dual regression
    pre_maps, post_maps = run_dual_regression(
        rsn_components, rsn_indices, masker_bold, masker_ica
    )

    # 6. Diagnóstico por sujeto (primer RSN)
    diagnostico_mapas(pre_maps[:, 0, :], post_maps[:, 0, :])

    # 7. Estadística pareada + guardar mapas
    resumen = paired_stats(pre_maps, post_maps, rsn_indices, masker_bold, gm_mask)

    print("\nRESUMEN ESTADÍSTICO:")
    print(resumen.to_string(index=False))

    # 8. Visualización
    print("\nGenerando figuras de resultados...")
    plot_results_v2(resumen, rsn_indices, masker_bold)

    print("\n" + "=" * 60)
    print("FASE 2 COMPLETADA")
    print(f"Resultados en: {STATS_DIR}")
    print("=" * 60)

    from scipy.stats import t as t_dist
    df_val   = len(SUBJECT_IDS) - 1
    T_THRESH_001 = float(t_dist.ppf(0.999, df=df_val))   # t para p<0.001 unilateral, df=19 → ~3.55
    T_THRESH_VIZ = 2.5  # umbral visual más estricto que 2.1 para limpiar ruido

    print(f"\n  Threshold p<.001 (df={df_val}): |t| > {T_THRESH_001:.2f}")

    n_rsn = len(rsn_indices)

    # ── Figura principal: glass brain con umbral p<.001 ──────────────────
    # Altura extra por fila: 0.35 para el label encima + 2.8 para el cerebro
    row_h = 3.3
    fig, axes = plt.subplots(n_rsn, 1,
                             figsize=(12, row_h * n_rsn),
                             squeeze=False)
    fig.patch.set_facecolor("black")

    for row_i, comp_idx in enumerate(rsn_indices):
        tmap_path = STATS_DIR / f"comp{comp_idx:02d}_tmap.nii.gz"
        row_data  = resumen[resumen["componente_ICA"] == comp_idx].iloc[0]
        max_t     = float(row_data["max_abs_t"])
        n_pos     = int(row_data["vox_post_gt_pre_p001"])
        n_neg     = int(row_data["vox_pre_gt_post_p001"])
        peak_mni  = plotting.find_xyz_cut_coords(
            nib.load(str(tmap_path)), activation_threshold=max_t * 0.9
        )
        peak_str  = f"[{peak_mni[0]:.1f}, {peak_mni[1]:.1f}, {peak_mni[2]:.1f}]"

        ax = axes[row_i][0]
        ax.set_facecolor("black")

        # Dibuja el glass brain SIN title (lo ponemos aparte)
        plotting.plot_glass_brain(
            tmap_path,
            display_mode="lyrz",
            colorbar=True,
            cmap="cold_hot",
            threshold=T_THRESH_001,
            vmax=max_t,
            title=None,               # ← sin título interno de nilearn
            axes=ax,
            black_bg=True,
            plot_abs=False,
        )

        # Título encima del eje, fuera del área del cerebro
        label = (
            f"comp#{comp_idx}  |  post>pre: {n_pos}vóx   pre>post: {n_neg}vóx  "
            f"|  max|t|={max_t:.2f}  |  pico MNI={peak_str}"
        )
        ax.set_title(label, fontsize=9, color="white", pad=4,
                     loc="left", fontweight="bold")

    plt.suptitle(
        f"Dual Regression — Δ(post−pre) | umbral p<.001 sin corrección (n={len(SUBJECT_IDS)})",
        fontsize=12, y=1.002, color="white"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.999])
    out = OUTPUT_DIR / "resultados_v2_glass_p001.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Figura glass brain guardada: {out}")

    # ── Figura 2: cortes axiales de los 3 componentes con más vóxeles p<.001 ──
    resumen_sorted = resumen.copy()
    resumen_sorted["total_p001"] = resumen_sorted["vox_post_gt_pre_p001"] + resumen_sorted["vox_pre_gt_post_p001"]
    top3 = resumen_sorted.nlargest(3, "total_p001")["componente_ICA"].tolist()

    fig2, axes2 = plt.subplots(len(top3), 1,
                               figsize=(14, 4.5 * len(top3)),
                               squeeze=False)

    for row_i, comp_idx in enumerate(top3):
        tmap_path = STATS_DIR / f"comp{comp_idx:02d}_tmap.nii.gz"
        row_data  = resumen[resumen["componente_ICA"] == comp_idx].iloc[0]
        max_t     = float(row_data["max_abs_t"])
        n_pos     = int(row_data["vox_post_gt_pre_p001"])
        n_neg     = int(row_data["vox_pre_gt_post_p001"])

        ax = axes2[row_i][0]
        plotting.plot_stat_map(
            tmap_path,
            display_mode="z",         # cortes axiales
            cut_coords=8,             # 8 cortes automáticos
            colorbar=True,
            cmap="cold_hot",
            threshold=T_THRESH_001,
            vmax=max_t,
            title=f"comp#{comp_idx}  |  post>pre={n_pos}vóx  pre>post={n_neg}vóx  |  max|t|={max_t:.2f}",
            axes=ax,
            black_bg=True,
        )

    plt.suptitle(
        "Top 3 componentes (más vóxeles p<.001) — cortes axiales",
        fontsize=13, y=1.005, color="white"
    )
    fig2.patch.set_facecolor("black")
    plt.tight_layout()
    out2 = OUTPUT_DIR / "resultados_v2_axial_top3.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Figura axial top3 guardada: {out2}")