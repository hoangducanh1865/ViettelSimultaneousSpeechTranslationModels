#!/usr/bin/env bash
# Thư viện dùng chung cho benchmark_qa.sh.
#
# Chịu trách nhiệm:
#   1. Nạp cấu hình từ .env (repo root rồi tới run/.env) -- đây là nơi DUY NHẤT đọc
#      HF_TOKEN/HF_WRITE_TOKEN, KHÔNG nhận 2 giá trị này qua CLI argument.
#   2. Suy ra toàn bộ đường dẫn (src/, Drive data, output từng task) từ REPO_DIR +
#      LOCATION (drive|local), cho phép override bằng biến môi trường đã export trước.
#   3. Cung cấp helper log/run/require_file/die.
#
# KHÔNG được `set -e` ở đây để còn source được; benchmark_qa.sh bật `set -euo pipefail`.

# ---------------------------------------------------------------------------------------------
# Vị trí file: <repo>/src/test_set/datasets_qa/benchmark_qa/run/_lib.sh
# ---------------------------------------------------------------------------------------------
_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_BENCHMARK_QA_SRC_DIR="$(cd "${_RUN_DIR}/.." && pwd)"
DATASETS_QA_SRC_DIR="$(cd "${_BENCHMARK_QA_SRC_DIR}/.." && pwd)"
# DATASETS_QA_SRC_DIR = <repo>/src/test_set/datasets_qa -> SRC_DIR = <repo>/src (lên 2 cấp).
SRC_DIR="$(cd "${DATASETS_QA_SRC_DIR}/../.." && pwd)"
DEFAULT_REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------------------------
# Nạp .env (giá trị đã export trước đó vẫn thắng vì .env chỉ ghi đè biến shell). Thứ tự:
# repo root .env -> run/.env (file sau ghi đè file trước).
# ---------------------------------------------------------------------------------------------
_load_env_file() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    set -a
    # shellcheck disable=SC1090
    . "$f" || { echo "[_lib] CẢNH BÁO: không source được $f (bỏ qua)." >&2; }
    set +a
}
_load_env_file "${DEFAULT_REPO_DIR}/.env"
_load_env_file "${_RUN_DIR}/.env"

# ---------------------------------------------------------------------------------------------
# Mặc định chung (có thể override bằng flag --repo-dir/--location hoặc biến môi trường).
# ---------------------------------------------------------------------------------------------
: "${REPO_DIR:=${DEFAULT_REPO_DIR}}"
: "${LOCATION:=drive}"
: "${PYTHON:=python3}"
: "${DEBATE_MODE:=api}"                 # api (2 API tự debate) hoặc manual (copy-paste) -- 2 chế độ chạy Y HỆT các bước sau
: "${DRY_RUN:=0}"
# Model MẠNH dùng cho bước debate + 2 lượt lọc (có thể override bằng flag tương ứng).
: "${GEMINI_MODEL:=gemini-3.1-pro-preview}"
: "${OPENAI_MODEL:=cx/gpt-5.6-luna}"
GEMINI_BASE_URL="${GEMINI_BASE_URL:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_API_KEY_FILE="${OPENAI_API_KEY_FILE:-}"
ENV_FILE="${ENV_FILE:-}"

# ---------------------------------------------------------------------------------------------
# resolve_paths(): tính lại mọi đường dẫn. Gọi SAU khi parse xong flag để --location /
# --repo-dir / --qa-datasets-dir có hiệu lực. Dùng `:=` nên giá trị đã set (từ .env hoặc
# flag export sẵn) luôn được giữ, chỉ điền default cho cái còn thiếu.
# ---------------------------------------------------------------------------------------------
resolve_paths() {
    REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
    SRC_DIR="${REPO_DIR}/src"
    DATASETS_QA_SRC_DIR="${SRC_DIR}/test_set/datasets_qa"
    BENCHMARK_QA_SRC_DIR="${DATASETS_QA_SRC_DIR}/benchmark_qa"
    TRANSLATE_DATASETS_SRC_DIR="${DATASETS_QA_SRC_DIR}/translate_datasets"
    SOUND_SRC_DIR="${BENCHMARK_QA_SRC_DIR}/sound"
    HIEN_TUONG_SRC_DIR="${BENCHMARK_QA_SRC_DIR}/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
    HAN_VIET_SRC_DIR="${HIEN_TUONG_SRC_DIR}/han_viet"
    PHUONG_NGU_SRC_DIR="${HIEN_TUONG_SRC_DIR}/phuong_ngu"
    TU_MUON_SRC_DIR="${HIEN_TUONG_SRC_DIR}/tu_muon"
    TU_LAY_SRC_DIR="${HIEN_TUONG_SRC_DIR}/tu_lay"
    CODE_SWITCHING_SRC_DIR="${BENCHMARK_QA_SRC_DIR}/speech/trich_xuat_thong_tin/code_switching"
    FILTER_QA_PIPELINE="${HIEN_TUONG_SRC_DIR}/filter_qa_pipeline.py"
    TOOLS_DIR="${BENCHMARK_QA_SRC_DIR}/tools"
    HF_PR_PUSH="${TRANSLATE_DATASETS_SRC_DIR}/hf_pr_push.py"

    # PYTHONPATH trỏ vào src/ để `from test_set.datasets_qa...` import được.
    export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

    # --- Dữ liệu: drive dùng Drive, local dùng <repo>/data (KHÔNG còn stuff/) ---
    if [[ "${LOCATION}" == "local" ]]; then
        : "${QA_DATASETS_DIR:=${REPO_DIR}/data}"
        : "${ENV_FILE:=${REPO_DIR}/.env}"
    else
        : "${QA_DATASETS_DIR:=/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/datasets/public/qa_datasets}"
        : "${ENV_FILE:=/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/.env}"
    fi
    : "${SERVICE_ACCOUNT_JSON:=/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/gemini_service_account.json}"
    : "${HF_SPEECH_REPO:=anhnbd2005/Vietnamese-Speech-QA}"
    : "${HF_SOUND_REPO:=anhnbd2005/Vietnamese-Audio-QA}"

    BENCHMARK_QA_SPEECH_DIR="${QA_DATASETS_DIR}/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
    : "${FULL_TRANSCRIPTS_JSON_PATH:=${QA_DATASETS_DIR}/benchmark_qa/speech/full_transcripts.json}"
    : "${RELEASE_HF_TRANSCRIPTS_JSON_PATH:=${QA_DATASETS_DIR}/benchmark_qa/speech/release_hf_transcripts_by_dataset.json}"

    # --- Code-switching + knowledge dir dùng CÙNG layout tương đối cho cả drive lẫn local ---
    : "${CODE_SWITCHING_DIR:=${QA_DATASETS_DIR}/benchmark_qa/speech/trich_xuat_thong_tin/code_switching}"
    : "${KNOWLEDGE_DIR:=${QA_DATASETS_DIR}/benchmark_qa/knowledge}"

    # Export để env_paths.py (chạy trong process con) đọc được cùng override.
    export QA_DATASETS_DIR CODE_SWITCHING_DIR KNOWLEDGE_DIR ENV_FILE

    # --- Sound ---
    CLOTHO_AQA_DIR="${QA_DATASETS_DIR}/sound_datasets/clotho_aqa"
    CLOTHO_AQA_MANIFEST="${CLOTHO_AQA_DIR}/clotho_aqa_test.jsonl"
    CLOTHO_AQA_NON_YESNO_MANIFEST="${CLOTHO_AQA_DIR}/clotho_aqa_test_non_yesno.jsonl"
    CLOTHO_AQA_VN_MANIFEST="${CLOTHO_AQA_DIR}/clotho_aqa_test_non_yesno_vn.jsonl"
    CLOTHO_AQA_VN_FILTER_CACHE="${CLOTHO_AQA_DIR}/clotho_aqa_non_yesno_vn_relevance_cache.json"
    CLOTHO_AQA_VI_QA_OUTPUT="${CLOTHO_AQA_DIR}/clotho_aqa_test_vi_qa.jsonl"
    MMAU_TEST_MINI_PATH="${QA_DATASETS_DIR}/mmau_sound/sound-test-mini.jsonl"
    MMAU_TEST_MINI_AUDIO_DIR="${QA_DATASETS_DIR}/mmau_sound/audio"
    SOUND_OUT_DIR="${QA_DATASETS_DIR}/sound"
    TEST_SOUND_MANIFEST_PATH="${QA_DATASETS_DIR}/test_sound.jsonl"
    SOUND_ZIP_PATH="${QA_DATASETS_DIR}/sound.zip"

    # --- Vietnamese-Speech-QA (tải test_speech.jsonl + audio subset) ---
    : "${VIETNAMESE_SPEECH_QA_LOCAL_DIR:=${QA_DATASETS_DIR}/Vietnamese-Speech-QA}"
    TEST_SPEECH_JSONL="${VIETNAMESE_SPEECH_QA_LOCAL_DIR}/test_speech.jsonl"

    # --- 4 task "Hiện tượng đặc biệt" ---
    HAN_VIET_OUT_DIR="${BENCHMARK_QA_SPEECH_DIR}/han_viet"
    HAN_VIET_INPUT_PATH="${HAN_VIET_OUT_DIR}/han_viet_qa.jsonl"
    HAN_VIET_SEED_CSV="${HAN_VIET_OUT_DIR}/han_viet_seed.csv"
    HAN_VIET_COVERAGE_REPORT="${HAN_VIET_OUT_DIR}/word_coverage_report_han_viet.json"
    HAN_VIET_FINAL_CSV="${HAN_VIET_OUT_DIR}/han_viet_final.csv"
    HAN_VIET_DIFFICULTY_OUTPUT="${HAN_VIET_OUT_DIR}/han_viet_difficulty_levels.jsonl"
    HAN_VIET_NEW_WORD_SAMPLES="${HAN_VIET_OUT_DIR}/asr_samples_with_han_viet.json"
    HAN_VIET_MULTIHOP_OUTPUT="${HAN_VIET_OUT_DIR}/han_viet_multihop_qa.jsonl"
    HAN_VIET_FINAL_QA="${HAN_VIET_OUT_DIR}/han_viet_multihop_qa_final.jsonl"

    PHUONG_NGU_OUT_DIR="${BENCHMARK_QA_SPEECH_DIR}/phuong_ngu"
    PHUONG_NGU_INPUT_PATH="${PHUONG_NGU_OUT_DIR}/phuong_ngu_qa.jsonl"
    PHUONG_NGU_COVERAGE_REPORT="${PHUONG_NGU_OUT_DIR}/word_coverage_report_phuong_ngu.json"
    PHUONG_NGU_REGION_OUTPUT="${PHUONG_NGU_OUT_DIR}/phuong_ngu_region_labels.jsonl"
    PHUONG_NGU_QA_OUTPUT="${PHUONG_NGU_OUT_DIR}/phuong_ngu_region_qa.jsonl"
    PHUONG_NGU_FINAL_QA="${PHUONG_NGU_OUT_DIR}/phuong_ngu_region_qa_final.jsonl"

    TU_MUON_INPUT_DIR="${QA_DATASETS_DIR}/tu_muon"
    TU_MUON_SAMPLES_PATH="${TU_MUON_INPUT_DIR}/asr_samples_with_tu_muon.json"
    TU_MUON_CSV_PATH="${TU_MUON_INPUT_DIR}/tu_muon_tieng_viet_viet_hoa_final.csv"
    TU_MUON_FINAL_DIR="${BENCHMARK_QA_SPEECH_DIR}/tu_muon"
    TU_MUON_DIFFICULTY_OUTPUT="${TU_MUON_FINAL_DIR}/tu_muon_difficulty_levels.jsonl"
    TU_MUON_MULTIHOP_OUTPUT="${TU_MUON_FINAL_DIR}/tu_muon_multihop_qa.jsonl"
    TU_MUON_FINAL_QA="${TU_MUON_FINAL_DIR}/tu_muon_multihop_qa_final.jsonl"

    TU_LAY_INPUT_DIR="${QA_DATASETS_DIR}/tu_lay"
    TU_LAY_FINAL_DIR="${BENCHMARK_QA_SPEECH_DIR}/tu_lay"

    # --- Code-switching: output của 2 lượt lọc (filter-questions) + file final ---
    CS_MULTIHOP="${CODE_SWITCHING_DIR}/code_switching_multihop_qa.jsonl"
    CS_GEMINI_KEPT="${CODE_SWITCHING_DIR}/code_switching_gemini_kept.jsonl"
    CS_OPENAI_KEPT="${CODE_SWITCHING_DIR}/code_switching_openai_kept.jsonl"
    CS_OPENAI_KEPT_FINAL="${CODE_SWITCHING_DIR}/code_switching_openai_kept_final.jsonl"
}

# Từ 1 file pre-final -> các đường dẫn của bước lọc 2 API + file final.
# Set: FILTER_GEM_JSONL, FILTER_OAI_JSONL, FILTER_GEM_RULES, FILTER_OAI_RULES, FILTER_FINAL.
filtered_paths() {
    local pre="$1" final="$2"
    local dir stem
    dir="$(dirname "$pre")"
    stem="$(basename "${pre%.jsonl}")"
    FILTER_GEM_JSONL="${dir}/${stem}_gemini_kept.jsonl"
    FILTER_OAI_JSONL="${dir}/${stem}_openai_kept.jsonl"
    FILTER_GEM_RULES="${dir}/${stem}_gemini_rules.json"
    FILTER_OAI_RULES="${dir}/${stem}_openai_rules.json"
    FILTER_FINAL="${final}"
}

# Đường dẫn input/output của 1 variant từ láy (gọi sau resolve_paths).
tu_lay_variant_paths() {
    local variant="$1"
    case "$variant" in
        toan_bo)
            TU_LAY_VARIANT_SAMPLES="${TU_LAY_INPUT_DIR}/asr_samples_with_tu_lay_toan_bo.json"
            TU_LAY_VARIANT_CSV="${TU_LAY_INPUT_DIR}/tu_lay_toan_bo_final.csv"
            ;;
        van)
            TU_LAY_VARIANT_SAMPLES="${TU_LAY_INPUT_DIR}/asr_samples_with_tu_lay_van.json"
            TU_LAY_VARIANT_CSV="${TU_LAY_INPUT_DIR}/tu_lay_van_final.csv"
            ;;
        chung)
            TU_LAY_VARIANT_SAMPLES="${TU_LAY_INPUT_DIR}/asr_samples_with_tu_lay.json"
            TU_LAY_VARIANT_CSV="${TU_LAY_INPUT_DIR}/tu_lay_tieng_viet_final.csv"
            ;;
        *) die "Variant từ láy không hợp lệ: $variant (phải là toan_bo|van|chung)." ;;
    esac
    TU_LAY_VARIANT_DIFFICULTY="${TU_LAY_FINAL_DIR}/tu_lay_${variant}_difficulty_levels.jsonl"
    TU_LAY_VARIANT_MULTIHOP="${TU_LAY_FINAL_DIR}/tu_lay_${variant}_multihop_qa.jsonl"
    TU_LAY_VARIANT_FINAL_QA="${TU_LAY_FINAL_DIR}/tu_lay_${variant}_multihop_qa_final.jsonl"
}

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
log() { printf '\033[1;34m[benchmark_qa]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[benchmark_qa][cảnh báo]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[benchmark_qa][lỗi]\033[0m %s\n' "$*" >&2; exit 1; }

# In + chạy lệnh (tôn trọng DRY_RUN=1). Che các secret khi in ra log.
_print_command() {
    local a secret
    printf '+'
    for a in "$@"; do
        for secret in "${HF_TOKEN:-}" "${HF_WRITE_TOKEN:-}" "${OPENAI_API_KEY:-}" "${GEMINI_API_KEY:-}" "${COMMONVOICE_API_KEY:-}"; do
            [[ -n "$secret" && "$a" == "$secret" ]] && a="***"
        done
        printf ' %q' "$a"
    done
    printf '\n'
}
run() {
    _print_command "$@"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    "$@"
}

py() { run "${PYTHON}" "$@"; }

require_file() {
    local f="$1" label="${2:-input}"
    if [[ ! -f "$f" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then warn "[dry-run] không thấy ${label}: $f"; else die "Không tìm thấy ${label}: $f"; fi
    fi
}
require_dir() {
    local d="$1" label="${2:-thư mục}"
    if [[ ! -d "$d" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then warn "[dry-run] không thấy ${label}: $d"; else die "Không tìm thấy ${label}: $d"; fi
    fi
}
require_var() {
    local name="$1" value="$2"
    if [[ -z "$value" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            warn "[dry-run] thiếu biến $name -- bỏ qua kiểm tra."
        else
            die "Thiếu biến $name (đặt trong run/.env hoặc .env ở gốc repo, rồi chạy lại)."
        fi
    fi
}

# Đường dẫn service account Gemini (tùy chọn): nếu không có, pipeline tự fallback sang .env proxy.
# In ra stdout path nếu file tồn tại; rỗng nếu không (kèm cảnh báo ra stderr).
gemini_service_account_arg() {
    if [[ ! -f "${SERVICE_ACCOUNT_JSON}" ]]; then
        warn "không thấy service account Gemini (${SERVICE_ACCOUNT_JSON}) -- dùng .env: ${ENV_FILE}"
        return 0
    fi
    printf '%s' "${SERVICE_ACCOUNT_JSON}"
}

# Nạp HF_TOKEN / HF_WRITE_TOKEN chỉ từ .env, KHÔNG nhận qua CLI.
require_hf_read_token() { require_var "HF_TOKEN" "${HF_TOKEN:-}"; }
require_hf_write_token() { require_var "HF_WRITE_TOKEN" "${HF_WRITE_TOKEN:-}"; }
