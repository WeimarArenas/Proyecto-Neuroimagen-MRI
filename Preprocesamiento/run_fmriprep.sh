#!/bin/bash
#
# =============================================================================
#  fMRIPrep Preprocessing Script for BIDS Dataset
#  Dataset: Pre-Post rehabilitation fMRI data of post-stroke patients
#  DOI: 10.18112/openneuro.ds003999.v1.0.2
# =============================================================================
#
#  This script automates fMRIPrep execution for all subjects in a BIDS dataset
#  using Docker. It iterates over each sub-XX folder and launches fMRIPrep
#  individually, enabling reproducibility and standardization.
#
#  Usage:
#    chmod +x run_fmriprep.sh
#    ./run_fmriprep.sh
#
# =============================================================================

# ----------------------------- CONFIGURATION ---------------------------------

# Path to the BIDS dataset containing sub-XX folders
BIDS_DIR="/mnt/c/Users/ASUS/Documents/OpenNeuro/datos_originales"

# Folder where fMRIPrep results will be saved
OUTPUT_DIR="/mnt/c/Users/ASUS/Documents/OpenNeuro/preprocesamiento"

# Enable/disable FreeSurfer cortical surface reconstruction (0 = off, 1 = on)
# Note: FreeSurfer adds ~8-12 hours per subject. Disable if not needed.
FS_FLAG=0

# Path to the FreeSurfer license file (required even if FS_FLAG=0)
FS_LICENSE="/mnt/c/Users/ASUS/Documents/OpenNeuro/license.txt"

# fMRIPrep Docker image version
FMRIPREP_VERSION="24.1.1"

# Number of CPUs to allocate per subject
N_CPUS=8

# Maximum memory (GB) to allocate per subject
MEM_GB=16

# Output space(s) for normalization
OUTPUT_SPACES="MNI152NLin2009cAsym:res-2 anat"

# Working directory for intermediate files (can be large, ~10-20 GB per subject)
WORK_DIR="/mnt/c/Users/ASUS/Documents/OpenNeuro/work"

# ----------------------------- VALIDATION ------------------------------------

echo "============================================="
echo " fMRIPrep Preprocessing Pipeline"
echo "============================================="
echo ""
echo " BIDS Directory : ${BIDS_DIR}"
echo " Output Directory : ${OUTPUT_DIR}"
echo " FreeSurfer       : $([ ${FS_FLAG} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo " fMRIPrep Version : ${FMRIPREP_VERSION}"
echo " CPUs per subject : ${N_CPUS}"
echo " Memory limit     : ${MEM_GB} GB"
echo "============================================="
echo ""

# Check that BIDS directory exists
if [ ! -d "${BIDS_DIR}" ]; then
    echo "ERROR: BIDS directory not found: ${BIDS_DIR}"
    exit 1
fi

# Check that dataset_description.json exists (required for BIDS)
if [ ! -f "${BIDS_DIR}/dataset_description.json" ]; then
    echo "ERROR: dataset_description.json not found. Is this a valid BIDS dataset?"
    exit 1
fi

# Check FreeSurfer license
if [ ! -f "${FS_LICENSE}" ]; then
    echo "WARNING: FreeSurfer license not found at ${FS_LICENSE}"
    echo "         You can obtain one free at: https://surfer.nmr.mgh.harvard.edu/registration.html"
    if [ ${FS_FLAG} -eq 1 ]; then
        echo "ERROR: FreeSurfer is enabled but license file is missing. Exiting."
        exit 1
    fi
fi

# Create output and work directories if they don't exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}"

# ----------------------------- BUILD FS OPTION -------------------------------

if [ ${FS_FLAG} -eq 0 ]; then
    FS_OPTION="--fs-no-reconall"
    echo "INFO: FreeSurfer reconstruction DISABLED (--fs-no-reconall)"
else
    FS_OPTION=""
    echo "INFO: FreeSurfer reconstruction ENABLED"
fi

# ----------------------------- DETECT SUBJECTS -------------------------------

# Automatically detect all sub-XX directories in the BIDS folder
SUBJECTS=($(find "${BIDS_DIR}" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort))

TOTAL=${#SUBJECTS[@]}

if [ ${TOTAL} -eq 0 ]; then
    echo "ERROR: No subjects found in ${BIDS_DIR}"
    exit 1
fi

echo ""
echo "Found ${TOTAL} subjects: ${SUBJECTS[*]}"
echo ""

# ----------------------------- PROCESSING LOOP -------------------------------

FAILED=()
SUCCESS=()
START_TIME=$(date +%s)

for i in "${!SUBJECTS[@]}"; do
    SUB=${SUBJECTS[$i]}
    SUB_ID=${SUB#sub-}  # Remove "sub-" prefix for fMRIPrep --participant-label
    CURRENT=$((i + 1))

    echo "---------------------------------------------"
    echo " Processing ${SUB} (${CURRENT}/${TOTAL})"
    echo " Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "---------------------------------------------"

    docker run --rm -it \
        -v "${BIDS_DIR}":/data:ro \
        -v "${OUTPUT_DIR}":/out \
        -v "${WORK_DIR}":/work \
        -v "${FS_LICENSE}":/opt/freesurfer/license.txt:ro \
        nipreps/fmriprep:${FMRIPREP_VERSION} \
        /data /out participant \
        --participant-label "${SUB_ID}" \
        --output-spaces ${OUTPUT_SPACES} \
        --nprocs ${N_CPUS} \
        --mem-mb $((MEM_GB * 1024)) \
        --work-dir /work \
        --skip_bids_validation \
        --stop-on-first-crash \
        ${FS_OPTION}

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "SUCCESS: ${SUB} completed."
        SUCCESS+=("${SUB}")
    else
        echo "FAILED: ${SUB} encountered an error."
        FAILED+=("${SUB}")
    fi

    echo ""
done

# ----------------------------- SUMMARY ---------------------------------------

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo "============================================="
echo " PREPROCESSING COMPLETE"
echo "============================================="
echo " Total time     : ${ELAPSED} minutes"
echo " Successful     : ${#SUCCESS[@]}/${TOTAL}"
echo " Failed         : ${#FAILED[@]}/${TOTAL}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed subjects: ${FAILED[*]}"
fi

echo ""
echo " Results saved to: ${OUTPUT_DIR}"
echo "============================================="
