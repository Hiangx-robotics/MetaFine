#!/usr/bin/env bash
set -euo pipefail

# OFT eval: assemble_I (assembling_kits_metafine_letter, target_letter=I)
# Stages:
#   stage1     clean + DR (camera_pos, camera_rot, ambient_light) | scene=I, instr=I
#   stage_sem1 clean only | scene=I, instr=N (semantic mismatch)
#   stage_sem2 clean only | scene=N AND instr=N

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${SCRIPT_DIR}/eval_fixed.sh"

OUT_ROOT="/nat/demos/openvla/long"
SUMMARY_JSON="${OUT_ROOT}/summary_live.json"
SUMMARY_JSONL="${OUT_ROOT}/summary_live.jsonl"

N_EPISODES=10

CKPT_ASSEMBLE_I=${CKPT_ASSEMBLE_I:-"${SCRIPT_DIR}/runs/openvla-7b+assemble_i_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--40000_chkpt"}

INSTR_I="grasp the character I, and Insert it into the hole to complete METAFINE"
INSTR_N="grasp the character N, and Insert it into the hole to complete METAFINE"

CAMERA_POS_LEVELS="[0.03, 0.06, 0.12]"
CAMERA_ROT_LEVELS_DEG="[2.0, 6.0, 12.0]"
LIGHT_AMBIENT_DELTA_LEVELS="[0.10, 0.25, 0.40]"

usage() {
  cat <<'EOF'
Usage:
  bash core/policies/openvla-oft/run_eval_assemble_i.sh [--n-episodes N]
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-episodes) N_EPISODES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "${EVAL_SH}" ]]; then
  echo "[ERROR] eval_fixed.sh not found at: ${EVAL_SH}" >&2
  exit 1
fi
if [[ ! -d "${CKPT_ASSEMBLE_I}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT_ASSEMBLE_I}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  cat > "${SUMMARY_JSON}" <<'EOF'
{ "updated_at": "", "tasks": {} }
EOF
fi

run_eval_once() {
  local out_dir="$1"
  local enable_dr="$2"
  local instr="$3"
  local target_letter="$4"   # "" => default I

  local cmd_env=(
    OUTPUT_ROOT="${out_dir}"
    OPENVLA_SUMMARY_JSON="${out_dir}/_internal_summary.json"
    OPENVLA_SUMMARY_JSONL="${out_dir}/_internal_summary.jsonl"
    TASK_DESCRIPTION_OVERRIDE="${instr}"
  )
  if [[ -n "${target_letter}" ]]; then
    cmd_env+=(TARGET_LETTER="${target_letter}")
  fi
  if [[ "${enable_dr}" == "1" ]]; then
    cmd_env+=(
      ENABLE_DR_EVAL=1
      CAMERA_POS_LEVELS="${CAMERA_POS_LEVELS}"
      CAMERA_ROT_LEVELS_DEG="${CAMERA_ROT_LEVELS_DEG}"
      LIGHT_AMBIENT_DELTA_LEVELS="${LIGHT_AMBIENT_DELTA_LEVELS}"
    )
  else
    cmd_env+=(ENABLE_DR_EVAL=0)
  fi
  mkdir -p "${out_dir}"
  env "${cmd_env[@]}" bash "${EVAL_SH}" "${CKPT_ASSEMBLE_I}" "assembling_kits_metafine_letter" "${N_EPISODES}"
}

update_live_summary() {
  local task_id="$1"; local task_desc="$2"; local task_root="$3"
  local s1="$4"; local ss1="$5"; local ss2="$6"
  python - "${SUMMARY_JSON}" "${SUMMARY_JSONL}" "${task_id}" "${task_desc}" "${task_root}" "${s1}" "${ss1}" "${ss2}" <<'PY'
import json, os, sys
from datetime import datetime
summary_json, summary_jsonl, task_id, task_desc, task_root, s1, ss1, ss2 = sys.argv[1:9]
def lj(p):
    if not p or p == "__NONE__" or not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f: return json.load(f)
data = lj(summary_json) or {"updated_at": "", "tasks": {}}
now = datetime.now().isoformat(timespec="seconds")
entry = {
    "task_id": task_id, "description": task_desc, "task_root": task_root,
    "stage1_metrics": lj(s1), "stage_sem1_metrics": lj(ss1), "stage_sem2_metrics": lj(ss2),
    "stage1_metrics_path": s1 if s1 != "__NONE__" else None,
    "stage_sem1_metrics_path": ss1 if ss1 != "__NONE__" else None,
    "stage_sem2_metrics_path": ss2 if ss2 != "__NONE__" else None,
    "updated_at": now,
}
data.setdefault("tasks", {})[task_id] = entry
data["updated_at"] = now
with open(summary_json, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
with open(summary_jsonl, "a", encoding="utf-8") as f: f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

echo "Live summary: ${SUMMARY_JSON}"

TASK_ID="task_assemble_i"
TASK_DESC="OFT assemble_I | stage1 clean+DR (scene=I,instr=I) + stage_sem1 (scene=I,instr=N) + stage_sem2 (scene=N,instr=N)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE_SEM1_DIR="${TASK_ROOT}/stage_sem1_instrN_sceneI"
STAGE_SEM2_DIR="${TASK_ROOT}/stage_sem2_instrN_sceneN"
mkdir -p "${TASK_ROOT}"

# stage1: scene=I, instr=I, DR on
run_eval_once "${STAGE1_DIR}" "1" "${INSTR_I}" ""
# stage_sem1: scene=I, instr=N, clean only
run_eval_once "${STAGE_SEM1_DIR}" "0" "${INSTR_N}" "I"
# stage_sem2: scene=N, instr=N, clean only
run_eval_once "${STAGE_SEM2_DIR}" "0" "${INSTR_N}" "N"

update_live_summary \
  "${TASK_ID}" "${TASK_DESC}" "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "${STAGE_SEM1_DIR}/metrics_summary.json" \
  "${STAGE_SEM2_DIR}/metrics_summary.json"

echo "OFT assemble_I eval done."
echo "Summary JSON:  ${SUMMARY_JSON}"
echo "Summary JSONL: ${SUMMARY_JSONL}"
