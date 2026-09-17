#!/usr/bin/env bash
set -euo pipefail

BASE="/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC"

SOURCE_ROOT="/home/omego/Documents/Hybrid_Stego_Eva/yedroudj_main_results_v3_2_fused075_payload_020_lr0001_gamma03_seed20260605"

SEED="20260605"
FEATURE_KIND="srm"

N_ESTIMATORS="100"
SUBSPACE_DIM="2000"

source "${BASE}/venv/bin/activate"

mkdir -p "${BASE}"/{pairs,runs,features,logs}

for P in 010 020 030 040; do
  echo ""
  echo "============================================================"
  echo "Running maxSRM/SRM+EC for payload_${P}"
  echo "============================================================"

  PAYLOAD_DIR="${SOURCE_ROOT}/payload_${P}"

  if [ ! -d "${PAYLOAD_DIR}" ]; then
    echo "[ERROR] Missing payload directory: ${PAYLOAD_DIR}"
    echo "Create or symlink payload_${P} first, then rerun."
    exit 1
  fi

  PAIRS_CSV="${BASE}/pairs/pairs_payload_${P}.csv"
  OUT_DIR="${BASE}/runs/payload_${P}_seed${SEED}_${FEATURE_KIND}_ec"
  CACHE_DIR="${BASE}/features/payload_${P}_${FEATURE_KIND}"

  python "${BASE}/scripts/01_build_pairs_from_payload_tree.py" \
    --input-root "${SOURCE_ROOT}" \
    --payload-code "${P}" \
    --out-csv "${PAIRS_CSV}" \
    2>&1 | tee "${BASE}/logs/build_pairs_payload_${P}.log"

  python "${BASE}/scripts/02_maxsrm_ec_train_eval.py" \
    --pairs-csv "${PAIRS_CSV}" \
    --out-dir "${OUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --feature-kind "${FEATURE_KIND}" \
    --seed "${SEED}" \
    --n-estimators "${N_ESTIMATORS}" \
    --subspace-dim "${SUBSPACE_DIM}" \
    --train-split train \
    --eval-splits val,test \
    2>&1 | tee "${BASE}/logs/train_eval_payload_${P}_${FEATURE_KIND}_ec.log"

done

echo ""
echo "[OK] All payloads completed."
echo "Results are under:"
echo "${BASE}/runs"
