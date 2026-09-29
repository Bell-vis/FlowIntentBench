#!/usr/bin/env bash
# Reproduce the seven-model calibration sample using validated receipts only.
# Exit 3 means reviews completed but some applicable metrics remain intervals.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
sample_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$sample_repo_root"
sample_output="$1"
sample_python="${PYTHON:-python}"
if [[ -e "$sample_output" ]]; then
  echo "Output already exists; choose a new directory." >&2
  exit 2
fi
sample_artifacts="outputs/model_answer_evaluation_v4"
sample_bank="outputs/model_answer_evaluation/frozen_auxiliary_references.json"

"$sample_python" scripts/evaluate_model_answers.py \
  --answers outputs/model_metric_calibration/answers \
  --cases blunt_fin_o1_f1 aideas_pressure_heterogeneity_o2_f1 aideas_high_shear_region_o3_f1 rt_density_vertical_motion_o1_f2 \
  --trials 1 --evidence-bank "$sample_bank" \
  --reference-package "$sample_artifacts/reference_construction/sample_high_shear_v3.json" \
  --review-repairs "$sample_artifacts/sample_repairs_v15" \
  --replay-from "$sample_artifacts/seven_model_sample_audited_v24" \
  --output "$sample_output/base" --offline --max-api-calls 0

"$sample_python" scripts/extend_evaluation_references.py \
  --source "$sample_output/base" --cases rt_density_vertical_motion_o1_f2 \
  --reference-package "$sample_artifacts/reference_construction/common_fields_v4.json" \
  --evidence-bank "$sample_bank" \
  --replay-from "$sample_artifacts/seven_model_rt_extension_v6" \
  --output "$sample_output/common" --compact-new-requests --replay-identical-prompts \
  --max-output-tokens 4000 --offline --max-api-calls 0

"$sample_python" scripts/extend_evaluation_references.py \
  --source "$sample_output/base" --cases rt_density_vertical_motion_o1_f2 \
  --prior-extension "$sample_output/common" \
  --prior-reference-package "$sample_artifacts/reference_construction/common_fields_v4.json" \
  --reference-package "$sample_artifacts/reference_construction/conditional_fields_v3.json" \
  --evidence-bank "$sample_bank" \
  --replay-from "$sample_artifacts/seven_model_conditional_extension_v5" \
  --output "$sample_output/conditional" --compact-new-requests --replay-identical-prompts \
  --max-output-tokens 3000 --offline --max-api-calls 0

"$sample_python" scripts/audit_evaluation_sample.py \
  --report "$sample_output/conditional/reports/experiment_report.json" \
  --output "$sample_output/audit" --receipt-root "$sample_output" --require-identified
