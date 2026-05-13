"""
=============================================================================
 Nilearn Post-fMRIPrep Motor Cortex Pipeline — ALL SUBJECTS
 Dataset: Pre-Post rehabilitation fMRI data of post-stroke patients
 Steps: Spatial Smoothing → Denoising → Motor ROI Time Series Extraction
 
 Motor network ROIs (AAL atlas):
   - Precentral (M1) L/R
   - Supp_Motor_Area (SMA) L/R
   - Postcentral (S1) L/R
   - Paracentral_Lobule L/R
   - Cerebelum (all motor lobules) L/R
=============================================================================

 Usage:
   python run_conn_all.py
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets, maskers
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURATION ==================================

BIDS_DIR     = r'C:\Users\ASUS\Documents\OpenNeuro\datos_originales'
DERIVATIVES  = r'C:\Users\ASUS\Documents\OpenNeuro\preprocesamiento'
OUTPUT_DIR   = r'C:\Users\ASUS\Documents\OpenNeuro\conn'

# 23 subjects with left-hemisphere lesion
SUBJECTS = [
    'sub-00', 'sub-01', 'sub-11', 'sub-12', 'sub-13',
    'sub-15', 'sub-16', 'sub-17', 'sub-18',
    'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24',
    'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29',
    'sub-30', 'sub-33', 'sub-34', 'sub-35',
]

SESSIONS = ['ses-pre', 'ses-post']

# Processing parameters
FWHM      = 6        # Smoothing kernel (mm)
HIGH_PASS = 0.008    # Hz
LOW_PASS  = 0.09     # Hz
TR        = 3.0      # seconds

# Confound columns (24 motion + CSF + WM = 26 regressors)
CONFOUND_COLUMNS = [
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
    'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
    'rot_x_power2', 'rot_y_power2', 'rot_z_power2',
    'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2',
    'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2',
    'csf', 'white_matter',
]

# Motor network ROIs from AAL atlas (complete motor network)
MOTOR_ROI_LABELS = [
    'Precentral_L', 'Precentral_R',
    'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
    'Postcentral_L', 'Postcentral_R',
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
    'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R',
    'Cerebelum_3_L', 'Cerebelum_3_R',
    'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
    'Cerebelum_6_L', 'Cerebelum_6_R',
    'Cerebelum_7b_L', 'Cerebelum_7b_R',
    'Cerebelum_8_L', 'Cerebelum_8_R',
    'Cerebelum_9_L', 'Cerebelum_9_R',
    'Cerebelum_10_L', 'Cerebelum_10_R',
]

# ========================= FETCH ATLAS (once) ==============================

print("Loading AAL atlas...")
atlas = datasets.fetch_atlas_aal(version='SPM12')
atlas_img    = atlas.maps
atlas_labels = atlas.labels

# Build name → numeric index mapping for AAL atlas
atlas_name_to_idx = {name: int(idx) for name, idx in zip(atlas.labels, atlas.indices)}

# Verify all motor labels exist in the atlas
for roi in MOTOR_ROI_LABELS:
    if roi not in atlas_name_to_idx:
        raise ValueError(f"ROI '{roi}' not found in AAL atlas labels")
print(f"  AAL atlas: {len(atlas_labels)} total ROIs")
print(f"  Motor network: {len(MOTOR_ROI_LABELS)} ROIs selected\n")

# ========================= HELPER FUNCTIONS ================================

def locate_files(subject, ses):
    """Return paths for func, confounds and mask for one session."""
    base = os.path.join(DERIVATIVES, subject, ses, 'func')
    func = os.path.join(base,
        f'{subject}_{ses}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
    conf = os.path.join(base,
        f'{subject}_{ses}_task-rest_desc-confounds_timeseries.tsv')
    mask = os.path.join(base,
        f'{subject}_{ses}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz')
    return func, conf, mask


def process_subject(subject):
    """Pipeline: smooth → denoise → extract motor ROI time series."""
    sub_out = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(sub_out, exist_ok=True)

    # ------ Verify all files exist ------
    for ses in SESSIONS:
        func, conf, mask = locate_files(subject, ses)
        for label, f in [('Functional', func), ('Confounds', conf), ('Mask', mask)]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f'{label} ({ses}): {f}')

    for ses in SESSIONS:
        func, conf_path, mask = locate_files(subject, ses)

        # 1. Spatial smoothing
        smoothed = image.smooth_img(func, fwhm=FWHM)
        smooth_path = os.path.join(sub_out,
            f'{subject}_{ses}_task-rest_desc-smooth_bold.nii.gz')
        smoothed.to_filename(smooth_path)

        # 2. Denoising (confound regression + band-pass)
        conf_df = pd.read_csv(conf_path, sep='\t')
        available = [c for c in CONFOUND_COLUMNS if c in conf_df.columns]
        confounds_matrix = conf_df[available].fillna(0).values

        n_vols = smoothed.shape[3]
        min_vols_for_bandpass = 3 * (2 * 5 + 1) + 1  # 34 for order 5
        if n_vols < min_vols_for_bandpass:
            print(f"         NOTE: {ses} has only {n_vols} volumes — "
                  f"skipping bandpass (confound regression only)")
            cleaned = image.clean_img(
                smoothed,
                confounds=confounds_matrix,
                high_pass=None,
                low_pass=None,
                t_r=TR,
                standardize='zscore_sample',
                mask_img=mask,
            )
        else:
            cleaned = image.clean_img(
                smoothed,
                confounds=confounds_matrix,
                high_pass=HIGH_PASS,
                low_pass=LOW_PASS,
                t_r=TR,
                standardize='zscore_sample',
                mask_img=mask,
            )
        clean_path = os.path.join(sub_out,
            f'{subject}_{ses}_task-rest_desc-denoised_bold.nii.gz')
        cleaned.to_filename(clean_path)

        del smoothed

        # 3. Extract motor ROI time series (AAL — motor network only)
        roi_masker = maskers.NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=atlas_labels,
            standardize='zscore_sample',
            resampling_target='data',
            memory='nilearn_cache',
        )
        ts_all = roi_masker.fit_transform(cleaned)

        # masker.labels_ returns numeric indices (e.g. 2001, 2002...)
        # Map motor ROI names to their numeric indices, then find column positions
        fitted_labels = list(roi_masker.labels_)
        motor_indices = []
        motor_names   = []
        for roi_name in MOTOR_ROI_LABELS:
            numeric_idx = atlas_name_to_idx[roi_name]
            if numeric_idx in fitted_labels:
                col_pos = fitted_labels.index(numeric_idx)
                motor_indices.append(col_pos)
                motor_names.append(roi_name)

        ts_motor = ts_all[:, motor_indices]

        ts_df = pd.DataFrame(ts_motor, columns=motor_names)
        ts_df.to_csv(os.path.join(sub_out,
            f'{subject}_{ses}_task-rest_timeseries_motor_AAL.csv'), index=False)

        del cleaned

    return len(motor_names)


# ========================= MAIN LOOP ======================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start = time.time()
    results = []
    failed  = []

    print("=" * 70)
    print(f"  MOTOR CORTEX PIPELINE — {len(SUBJECTS)} subjects × {len(SESSIONS)} sessions")
    print(f"  Steps: Smoothing → Denoising → Motor ROI Time Series")
    print("=" * 70)
    print()

    for i, subj in enumerate(SUBJECTS, 1):
        t0 = time.time()

        # Skip subjects that already have all expected output files with valid content
        sub_out = os.path.join(OUTPUT_DIR, subj)
        expected_files = []
        for ses in SESSIONS:
            expected_files += [
                f'{subj}_{ses}_task-rest_desc-smooth_bold.nii.gz',
                f'{subj}_{ses}_task-rest_desc-denoised_bold.nii.gz',
                f'{subj}_{ses}_task-rest_timeseries_motor_AAL.csv',
            ]
        all_exist = all(os.path.isfile(os.path.join(sub_out, f)) for f in expected_files)
        # Also check that motor CSV has actual ROI columns (not empty)
        if all_exist:
            check_csv = os.path.join(sub_out,
                f'{subj}_ses-pre_task-rest_timeseries_motor_AAL.csv')
            try:
                csv_cols = pd.read_csv(check_csv, nrows=0).columns
                if len(csv_cols) < len(MOTOR_ROI_LABELS):
                    all_exist = False  # Force reprocessing
            except Exception:
                all_exist = False  # Force reprocessing
        if all_exist:
            print(f"[{i:02d}/{len(SUBJECTS)}] {subj} ... SKIPPED (already complete)")
            continue

        print(f"[{i:02d}/{len(SUBJECTS)}] {subj} ...", flush=True)

        try:
            n_rois = process_subject(subj)
            elapsed = time.time() - t0
            print(f"         done in {elapsed/60:.1f} min  |  "
                  f"{n_rois} motor ROIs extracted")
            results.append({
                'subject': subj,
                'n_motor_rois': n_rois,
                'time_min': elapsed / 60,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"         FAILED after {elapsed/60:.1f} min — {e}")
            failed.append(subj)

    total_elapsed = time.time() - total_start

    # ========================= GROUP SUMMARY ==================================

    print()
    print("=" * 70)
    print("  GROUP SUMMARY")
    print("=" * 70)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'group_motor_summary.csv'), index=False)

        print(f"  Subjects processed: {len(results)}/{len(SUBJECTS)}")
        print(f"  Motor ROIs per subject: {df['n_motor_rois'].iloc[0]}")
        print(f"  Avg time per subject:   {df['time_min'].mean():.1f} min")
        print(f"  Total elapsed:          {total_elapsed/60:.1f} min "
              f"({total_elapsed/3600:.1f} h)")

    if failed:
        print(f"\n  FAILED subjects: {', '.join(failed)}")

    print()
    print(f"  Results → {OUTPUT_DIR}")
    print("=" * 70)
