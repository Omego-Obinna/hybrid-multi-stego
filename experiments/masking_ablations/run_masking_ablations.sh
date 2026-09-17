#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/omego/Documents/Hybrid_Stego_Eva"
PKG="${ROOT}/masking_ablations"
OUT="${ROOT}/masking_ablations_outputs"
MODE="${1:-smoke}"

case "${MODE}" in
  audit)
    python "${PKG}/00_audit_qim_mask_layer.py" \
      --project-root "${ROOT}" \
      --out-dir "${OUT}/audit"
    ;;

  smoke)
    : "${QIM_MASK_ABLATION_KEY:?Set QIM_MASK_ABLATION_KEY to a benchmark-only 64-hex-character key}"
    python "${PKG}/01_generate_qim_mask_ablations.py" \
      --cover-root "/home/omego/Downloads/BOSSbase/BOSSbase_1.01" \
      --out-dir "${OUT}/smoke/generated" \
      --payloads "0.10" \
      --seeds "42,43" \
      --covers 5 \
      --psi-modes "hashid,global,block" \
      --delta-scales "0.5,1.0,2.0" \
      --base-delta 0.25 \
      --message-classes "zero,alternating,lowentropy,text,random" \
      --window-bits 2048 \
      --windows-per-sample 2 \
      --nist-max-bits 200000

    python "${PKG}/02_analyze_qim_mask_ablations.py" \
      --metrics-csv "${OUT}/smoke/generated/mask_sample_metrics.csv" \
      --features-csv "${OUT}/smoke/generated/mask_classifier_features.csv" \
      --out-dir "${OUT}/smoke/analysis" \
      --formats pdf png svg \
      --dpi 300
    ;;

  paper)
    : "${QIM_MASK_ABLATION_KEY:?Set QIM_MASK_ABLATION_KEY to a benchmark-only 64-hex-character key}"
    python "${PKG}/01_generate_qim_mask_ablations.py" \
      --cover-root "/home/omego/Downloads/BOSSbase/BOSSbase_1.01" \
      --out-dir "${OUT}/paper/generated" \
      --payloads "0.10,0.20,0.40" \
      --seeds "42,43,44" \
      --covers 50 \
      --psi-modes "hashid,global,block" \
      --delta-scales "0.5,1.0,2.0" \
      --base-delta 0.25 \
      --message-classes "zero,alternating,lowentropy,text,random" \
      --window-bits 2048 \
      --windows-per-sample 4 \
      --nist-max-bits 2000000 \
      --nist-message-classes "zero,text,random"

    python "${PKG}/02_analyze_qim_mask_ablations.py" \
      --metrics-csv "${OUT}/paper/generated/mask_sample_metrics.csv" \
      --features-csv "${OUT}/paper/generated/mask_classifier_features.csv" \
      --out-dir "${OUT}/paper/analysis" \
      --formats pdf png svg \
      --dpi 300

    python "${PKG}/03_summarise_existing_qim_recovery.py" \
      --log-root "${ROOT}/ours_qim_v3_2_fused075_payloads_FULL" \
      --out-dir "${OUT}/paper/recovery"
    ;;

  nist)
    : "${NIST_CONFIG_ID:?Set NIST_CONFIG_ID, e.g. cfg_04}"
    : "${NIST_PAYLOAD:?Set NIST_PAYLOAD, e.g. 0.10}"
    python "${PKG}/04_export_nist_sts_ascii.py" \
      --manifest "${OUT}/paper/generated/bitstream_manifest.csv" \
      --out-dir "${OUT}/paper/nist_ascii" \
      --config-id "${NIST_CONFIG_ID}" \
      --payload "${NIST_PAYLOAD}" \
      --message-class "${NIST_MESSAGE_CLASS:-zero}" \
      --max-files 30
    ;;

  *)
    echo "Usage: $0 {audit|smoke|paper|nist}" >&2
    exit 2
    ;;
esac
