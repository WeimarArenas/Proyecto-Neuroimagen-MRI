"""
=======================================================================
FASE 1 — ICA DE GRUPO + VISUALIZACIÓN INTERACTIVA DE COMPONENTES
=======================================================================

Corre primero este script. Al terminar:
  1. Se genera una figura con TODOS los componentes
  2. Se abre un visualizador interactivo componente por componente
  3. Tú ingresas cuáles son RSNs válidos
  4. Se guarda un archivo JSON con tu selección

Ese JSON lo usa el script fase2_dual_regression.py

Estructura CONN esperada:
  conn/
    sub-00/
      sub-00_ses-pre_task-rest_desc-denoised_bold.nii.gz
      sub-00_ses-post_task-rest_desc-denoised_bold.nii.gz
=======================================================================
"""

import numpy as np
import nibabel as nib
import json
import hashlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from pathlib import Path
from nilearn.decomposition import CanICA, DictLearning
from nilearn.image import index_img, math_img
from nilearn import plotting
import warnings
warnings.filterwarnings("ignore")


# =======================================================================
# CONFIGURACIÓN
# =======================================================================

CONN_DIR     = Path(r"C:\Users\ASUS\Desktop\OpenNeuro\conn")
OUTPUT_DIR   = Path(r"C:\Users\ASUS\Desktop\OpenNeuro\ica\resultados_ica")

# Los sujetos se detectarán automáticamente de las carpetas en CONN_DIR
# que empiecen con "sub-" y tengan los archivos requeridos.
SUBJECT_IDS  = []  # Se llenará automáticamente en check_files()

TASK         = "rest"
TR           = 3.0
N_COMPONENTS = 20
ICA_METHOD   = "canica"   # "canica" o "dictlearning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COHORT_TAG se definirá después de detectar sujetos válidos

# =======================================================================
# UTILIDADES
# =======================================================================

def get_bold_path(subject: str, session: str) -> Path:
    return (
        CONN_DIR / subject /
        f"{subject}_{session}_task-{TASK}_desc-denoised_bold.nii.gz"
    )

def check_files():
    global SUBJECT_IDS
    MIN_VOLUMES = 163  # mínimo de volúmenes por sesión para dual regression estable
    print("Detectando sujetos automáticamente y verificando archivos...")
    
    # Detectar todas las carpetas que empiecen con "sub-"
    all_subjects = [d.name for d in CONN_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    print(f"Encontrados {len(all_subjects)} sujetos potenciales: {all_subjects}")
    
    valid_subjects = []
    ref_shape = None  # se fija con el primer sujeto válido

    for sub in all_subjects:
        ok = True
        for ses in ["ses-pre", "ses-post"]:
            p = get_bold_path(sub, ses)
            if not p.exists():
                print(f"  EXCLUIDO {sub}: falta {p.name}")
                ok = False
                break
            img = nib.load(str(p))
            # Verificar shape espacial consistente
            shape3d = img.shape[:3]
            if ref_shape is None:
                ref_shape = shape3d
            elif shape3d != ref_shape:
                print(f"  EXCLUIDO {sub}: shape {shape3d} != referencia {ref_shape}")
                ok = False
                break
            # Verificar volúmenes mínimos (evitar sistema sobredeterminado en dual regression)
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

def build_common_mask() -> nib.Nifti1Image:
    from nilearn.image import new_img_like
    print("Construyendo máscara común...")
    mascara = None
    for sub in SUBJECT_IDS:
        for ses in ["ses-pre", "ses-post"]:
            bold = nib.load(str(get_bold_path(sub, ses)))
            data = bold.get_fdata()
            mask_sub = (data.std(axis=-1) > 0).astype(np.float32)
            img_mask = new_img_like(bold, mask_sub)
            mascara = img_mask if mascara is None else math_img("a * b", a=mascara, b=img_mask)
            mascara.to_filename(MASK_OUTPUT)
    print(f"  Máscara: {int(mascara.get_fdata().sum())} vóxeles\n")
    return mascara


# =======================================================================
# 1. ICA DE GRUPO
# =======================================================================

def run_group_ica(mask_img: nib.Nifti1Image):
    """Retorna (components_img, decomposer | None).
    decomposer es None si se cargó desde disco (ya calculado)."""
    # Si ya existe, cargarlo directamente
    if ICA_OUTPUT.exists():
        print(f"ICA encontrado en disco, cargando: {ICA_OUTPUT}")
        return nib.load(str(ICA_OUTPUT)), None

    print(f"Corriendo ICA de grupo ({ICA_METHOD}, {N_COMPONENTS} componentes)...")
    all_imgs = [str(get_bold_path(s, ses))
                for s in SUBJECT_IDS for ses in ["ses-pre", "ses-post"]]

    params = dict(
        n_components=N_COMPONENTS, mask=mask_img,
        smoothing_fwhm=6, t_r=TR,   # 6mm FWHM → mejora solapamiento espacial entre sujetos
        high_pass=None, low_pass=None,
        detrend=True, standardize=True,
        memory="nilearn_cache", memory_level=2,
        random_state=42, verbose=1,
    )
    decomposer = CanICA(**params) if ICA_METHOD == "canica" else DictLearning(**params)
    decomposer.fit(all_imgs)

    components_img = decomposer.components_img_
    components_img.to_filename(str(ICA_OUTPUT))
    print(f"  Guardado: {ICA_OUTPUT}\n")
    return components_img, decomposer


# =======================================================================
# 2. VISUALIZACIÓN — Figura resumen con todos los componentes
# =======================================================================

def plot_all_components(components_img: nib.Nifti1Image):
    """
    Genera y guarda una figura PNG con todos los componentes
    en vista axial. Útil para una revisión rápida general.
    """
    n = components_img.shape[-1]
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5, n_rows * 3))
    axes = axes.ravel()

    for i in range(n):
        plotting.plot_glass_brain(
            index_img(components_img, i),
            display_mode="z",
            colorbar=False,
            title=f"Comp #{i}",
            axes=axes[i],
            black_bg=False,
        )
    # Ocultar ejes sobrantes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Componentes ICA de grupo — selecciona los RSNs válidos",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out = OUTPUT_DIR / "todos_los_componentes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigura resumen guardada: {out}")
    print("Ábrela para tener una vista general de todos los componentes.\n")


# =======================================================================
# 3. VISUALIZACIÓN INTERACTIVA — Componente por componente
#    Navega con botones, marca como RSN o Ruido, guarda selección
# =======================================================================

def interactive_component_review(components_img: nib.Nifti1Image) -> list:
    """
    Abre una ventana interactiva donde puedes navegar componente por componente.

    Controles:
      [← Anterior]  [Siguiente →]  — navegar
      [✓ RSN]                      — marcar como red neuronal válida
      [✗ Ruido]                    — marcar como artefacto
      [Guardar y salir]            — terminar y guardar selección

    Retorna lista de índices de componentes marcados como RSN.
    """
    n = components_img.shape[-1]
    state = {
        "idx"     : 0,
        "labels"  : {i: None for i in range(n)},  # None / "RSN" / "Ruido"
        "running" : True,
    }

    matplotlib.use("TkAgg")   # backend interactivo

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    # Layout: vista cerebro arriba, controles abajo
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.3)
    ax_brain  = fig.add_subplot(gs[0])
    ax_ctrl   = fig.add_subplot(gs[1])
    ax_ctrl.axis("off")

    # Botones
    ax_prev  = fig.add_axes([0.05, 0.06, 0.12, 0.06])
    ax_next  = fig.add_axes([0.19, 0.06, 0.12, 0.06])
    ax_rsn   = fig.add_axes([0.37, 0.06, 0.14, 0.06])
    ax_noise = fig.add_axes([0.53, 0.06, 0.14, 0.06])
    ax_save  = fig.add_axes([0.74, 0.06, 0.20, 0.06])

    btn_prev  = Button(ax_prev,  "← Anterior",  color="lightgray")
    btn_next  = Button(ax_next,  "Siguiente →",  color="lightgray")
    btn_rsn   = Button(ax_rsn,   "✓ RSN",        color="#c8e6c9")
    btn_noise = Button(ax_noise, "✗ Ruido",      color="#ffcdd2")
    btn_save  = Button(ax_save,  "Guardar y salir", color="#bbdefb")

    status_text = fig.text(0.5, 0.015, "", ha="center", fontsize=10, color="gray")

    def get_label_color(label):
        return {"RSN": "#2e7d32", "Ruido": "#c62828", None: "#555555"}[label]

    def draw_component():
        ax_brain.clear()
        idx = state["idx"]
        comp_img = index_img(components_img, idx)
        label    = state["labels"][idx]
        label_str = f"  [{label}]" if label else "  [sin clasificar]"
        color_str = get_label_color(label)

        plotting.plot_glass_brain(
            comp_img,
            display_mode="lyrz",
            colorbar=True,
            title=f"Componente #{idx}{label_str}",
            axes=ax_brain,
            black_bg=False,
        )
        ax_brain.set_title(
            f"Componente #{idx} / {n-1}{label_str}",
            fontsize=13, color=color_str, pad=10
        )

        # Contador de clasificados
        clasificados = sum(1 for v in state["labels"].values() if v is not None)
        rsns = sum(1 for v in state["labels"].values() if v == "RSN")
        status_text.set_text(
            f"Clasificados: {clasificados}/{n}  |  RSNs marcados: {rsns}"
        )
        fig.canvas.draw_idle()

    def on_prev(event):
        state["idx"] = max(0, state["idx"] - 1)
        draw_component()

    def on_next(event):
        state["idx"] = min(n - 1, state["idx"] + 1)
        draw_component()

    def on_rsn(event):
        state["labels"][state["idx"]] = "RSN"
        draw_component()
        # Avanzar automáticamente al siguiente
        if state["idx"] < n - 1:
            state["idx"] += 1
            draw_component()

    def on_noise(event):
        state["labels"][state["idx"]] = "Ruido"
        draw_component()
        if state["idx"] < n - 1:
            state["idx"] += 1
            draw_component()

    def on_save(event):
        state["running"] = False
        plt.close(fig)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_rsn.on_clicked(on_rsn)
    btn_noise.on_clicked(on_noise)
    btn_save.on_clicked(on_save)

    draw_component()
    plt.show(block=True)

    # Recolectar índices RSN
    rsn_indices = sorted([i for i, v in state["labels"].items() if v == "RSN"])
    return rsn_indices


# =======================================================================
# 4. GUARDAR SELECCIÓN
# =======================================================================

def save_selection(rsn_indices: list, components_img: nib.Nifti1Image):
    """
    Guarda la selección en JSON y genera una figura solo con los RSNs elegidos.
    """
    n_total = components_img.shape[-1]
    selection = {
        "rsn_indices"   : rsn_indices,
        "n_total"       : n_total,
        "n_rsn"         : len(rsn_indices),
        "ica_method"    : ICA_METHOD,
        "n_components"  : N_COMPONENTS,
        "ica_file"      : str(ICA_OUTPUT),
        "mask_file"     : str(MASK_OUTPUT),
    }

    with open(SELECTION_FILE, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSelección guardada: {SELECTION_FILE}")
    print(f"RSNs seleccionados: {rsn_indices}")

    # Figura solo con los RSNs seleccionados
    if rsn_indices:
        n_rsn  = len(rsn_indices)
        n_cols = min(3, n_rsn)
        n_rows = int(np.ceil(n_rsn / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5, n_rows * 3.5),
                                 squeeze=False)
        axes = axes.ravel()

        for plot_i, comp_i in enumerate(rsn_indices):
            plotting.plot_glass_brain(
                index_img(components_img, comp_i),
                display_mode="lyrz",
                colorbar=True,
                title=f"RSN — Comp #{comp_i}",
                axes=axes[plot_i],
            )
        for j in range(n_rsn, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"RSNs seleccionados ({n_rsn} de {n_total})",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        out = OUTPUT_DIR / "RSNs_seleccionados.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figura RSNs guardada: {out}")


# =======================================================================
# MAIN
# =======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 1 — ICA DE GRUPO + SELECCIÓN DE COMPONENTES")
    print("=" * 60 + "\n")

    check_files()
    
    COHORT_TAG = f"{len(SUBJECT_IDS)}subs_{hashlib.sha1(','.join(SUBJECT_IDS).encode('utf-8')).hexdigest()[:8]}"
    
    # Archivo donde se guardará tu selección de componentes
    SELECTION_FILE = OUTPUT_DIR / "componentes_seleccionados.json"
    # Archivo donde se guardan los mapas ICA (para no recalcular en fase 2)
    ICA_OUTPUT     = OUTPUT_DIR / f"group_ICA_components_{COHORT_TAG}.nii.gz"
    MASK_OUTPUT    = OUTPUT_DIR / f"mascara_comun_{COHORT_TAG}.nii.gz"
    
    mask_img                    = build_common_mask()
    components_img, decomposer  = run_group_ica(mask_img)

    # Figura resumen (todos los componentes en una imagen)
    print("Generando figura resumen de todos los componentes...")
    plot_all_components(components_img)

    # Figura prob-atlas (contornos de todos los componentes superpuestos)
    print("Generando figura prob-atlas...")
    fig_pa, ax_pa = plt.subplots(1, 1, figsize=(16, 4))
    plotting.plot_prob_atlas(
        components_img,
        view_type="contours",
        axes=ax_pa,
        title="Todos los componentes ICA (prob atlas)",
        black_bg=True,
    )
    out_pa = OUTPUT_DIR / "todos_los_componentes_probatlas.png"
    plt.savefig(out_pa, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  Figura prob-atlas guardada: {out_pa}")

    # HTML report (solo si el ICA fue recalculado, no si se cargó de disco)
    if decomposer is not None:
        print("Generando reporte HTML...")
        try:
            report = decomposer.generate_report()
            html_out = OUTPUT_DIR / "reporte_ICA.html"
            report.save_as_html(str(html_out))
            print(f"  Reporte HTML guardado: {html_out}")
        except Exception as e:
            print(f"  [AVISO] No se pudo generar el HTML: {e}")

    # Revisor interactivo
    print("Abriendo visualizador interactivo...")
    print("  → Marca cada componente como RSN o Ruido")
    print("  → Cuando termines haz clic en 'Guardar y salir'\n")

    rsn_indices = interactive_component_review(components_img)

    if not rsn_indices:
        print("\nNo seleccionaste ningún componente. Revisa y vuelve a correr.")
    else:
        save_selection(rsn_indices, components_img)
        print(f"\nAhora corre: fase2_dual_regression.py")

    print("\n" + "=" * 60)
