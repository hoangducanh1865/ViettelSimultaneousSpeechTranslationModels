#!/usr/bin/env bash
# =============================================================================================
# benchmark_qa.sh -- dispatcher duy nhất cho toàn bộ pipeline xây benchmark QA
# (trước đây nằm rải rác trong notebook benchmark_qa_constructer copy 3.ipynb).
#
# Toàn bộ code orchestration đã được gom vào đây; notebook chỉ cần 3 cell:
#   mount Drive -> clone/pull repo -> `bash .../run/benchmark_qa.sh all --location drive`.
#
# Cấu hình (HF_TOKEN, HF_WRITE_TOKEN, ...) đọc từ `.env` (repo root hoặc run/.env), KHÔNG
# truyền token qua CLI argument. Xem `--help`.
# =============================================================================================
set -euo pipefail

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "${_RUN_DIR}/_lib.sh"

# ---------------------------------------------------------------------------------------------
# Default cho các tham số dùng chung (parse_common sẽ override nếu có flag).
# ---------------------------------------------------------------------------------------------
: "${TASK:=}"
: "${VARIANT:=}"
: "${PROVIDER:=}"
: "${TARGET:=all}"
: "${ON_MISSING:=skip}"
: "${PUSH:=0}"
: "${WITH_DEBATE:=0}"
: "${MAX_ROUNDS:=}"
: "${BATCH_SIZE:=}"
: "${MAX_WORKERS:=}"
: "${MAX_RETRIES:=}"
: "${MODEL:=}"
: "${SEED:=}"
: "${SKIP_SOUND:=0}"
: "${SKIP_HAN_VIET:=0}"
: "${SKIP_PHUONG_NGU:=0}"
: "${SKIP_TU_MUON:=0}"
: "${SKIP_TU_LAY:=0}"
: "${SKIP_FINALIZE:=0}"
: "${SKIP_PUSH:=0}"
: "${_INITIALIZED:=0}"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
benchmark_qa.sh -- pipeline xây benchmark QA (sound + 4 task "hiện tượng đặc biệt" + code-switching)

CÁCH DÙNG:
    bash run/benchmark_qa.sh <subcommand> [flags]

SUBCOMMAND:
    fetch-speech          Tải MAPPING_REPORT.json + test_speech.jsonl (và audio subset nếu chọn)
    fetch-speech-zip      Tải + giải nén speech.zip đã publish
    sound                 ClothoAQA: build-manifest -> lọc VN -> dịch -> merge MMAU (--push để PR)
    debate                Tri thức nền: build_debate_seed -> auto_model_relay (--task BẮT BUỘC)
    han-viet              Hán Việt: seed csv -> coverage -> classify -> KG -> samples -> generate -> finalize
    phuong-ngu            Phương ngữ: classify-region -> generate-questions -> finalize
    tu-muon               Từ mượn: classify-levels -> generate-questions -> finalize
    tu-lay                Từ láy: 3 variant (+ cloze cho toan_bo) -> finalize
    finalize              Chỉ chạy finalize_qa.py cho các file pre-final hiện có
    push-hf               Đẩy file final lên HF (--target speech|sound|all)
    code-switching        Trích xuất thông tin: scan-dictionary -> merge -> debate -> classify -> generate -> filter x2
    code-switching-mmsu   Nhánh MMSU: build-manifest subset Code-switching (111 sample)
    inspect               Thống kê/in manifest: <tree|stats|sample|head|knowledge-status>
    local-preprocess      Tiền xử lý local-only (--task tu-muon|tu-lay), cần stuff/benchmark_qa/
    all                   Chạy toàn bộ luồng (có thể skip từng phần)

FLAG DÙNG CHUNG:
    --location drive|local            (mặc định: drive)
    --repo-dir DIR                    Gốc repo (mặc định: tự suy từ vị trí script)
    --qa-datasets-dir DIR             Gốc dữ liệu QA trên Drive
    --service-account-json FILE       Service account Gemini (mặc định: Drive)
    --knowledge-dir DIR               Thư mục knowledge_<task>.json
    --full-transcripts-json FILE      Corpus text rộng (full_transcripts.json)
    --release-hf-transcripts-json F   Corpus CÓ audio (release_hf_transcripts_by_dataset.json)
    --speech-local-dir DIR            local_dir của Vietnamese-Speech-QA
    --python BIN                      (mặc định: python3)
    --task NAME                       (debate / local-preprocess)
    --variant toan_bo|van|chung|all   (tu-lay; mặc định: all)
    --provider gemini|openai          (code-switching filter-questions)
    --target speech|sound|all         (push-hf; mặc định: all)
    --on-missing skip|raise           (finalize; mặc định: skip)
    --model NAME --batch-size N --max-workers N --max-retries N --max-rounds N --seed N
    --push                            sound: zip + tạo PR lên HF
    --with-debate                     all: chạy thêm bước debate
    --skip-sound --skip-han-viet --skip-phuong-ngu --skip-tu-muon --skip-tu-lay
    --skip-finalize --skip-push       all: bỏ qua phần tương ứng
    --dry-run                         In lệnh sẽ chạy, không thực thi
    -h|--help

TOKEN (BẮT BUỘC ĐẶT TRONG .env, KHÔNG truyền qua flag):
    HF_TOKEN           đọc dữ liệu HF (fetch-speech, fetch-speech-zip)
    HF_WRITE_TOKEN     tạo PR lên HF (sound --push, push-hf)

VÍ DỤ:
    bash run/benchmark_qa.sh all --location drive
    bash run/benchmark_qa.sh sound --location drive --push
    bash run/benchmark_qa.sh debate --task code_switching --location local --debate-mode manual
    bash run/benchmark_qa.sh tu-lay --variant toan_bo --location drive
    bash run/benchmark_qa.sh inspect tree /path/test_sound.jsonl
    bash run/benchmark_qa.sh inspect knowledge-status
EOF
}

parse_common() {
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --location)                    LOCATION="$2"; shift 2 ;;
            --debate-mode)                 DEBATE_MODE="$2"; shift 2 ;;
            --repo-dir)                    REPO_DIR="$2"; shift 2 ;;
            --qa-datasets-dir)             QA_DATASETS_DIR="$2"; shift 2 ;;
            --service-account-json)        SERVICE_ACCOUNT_JSON="$2"; shift 2 ;;
            --knowledge-dir)               KNOWLEDGE_DIR="$2"; shift 2 ;;
            --full-transcripts-json)       FULL_TRANSCRIPTS_JSON_PATH="$2"; shift 2 ;;
            --release-hf-transcripts-json) RELEASE_HF_TRANSCRIPTS_JSON_PATH="$2"; shift 2 ;;
            --speech-local-dir)            VIETNAMESE_SPEECH_QA_LOCAL_DIR="$2"; shift 2 ;;
            --python)                      PYTHON="$2"; shift 2 ;;
            --task)                        TASK="$2"; shift 2 ;;
            --variant)                     VARIANT="$2"; shift 2 ;;
            --provider)                    PROVIDER="$2"; shift 2 ;;
            --target)                      TARGET="$2"; shift 2 ;;
            --model)                       MODEL="$2"; shift 2 ;;
            --batch-size)                  BATCH_SIZE="$2"; shift 2 ;;
            --max-workers)                 MAX_WORKERS="$2"; shift 2 ;;
            --max-retries)                 MAX_RETRIES="$2"; shift 2 ;;
            --max-rounds)                  MAX_ROUNDS="$2"; shift 2 ;;
            --seed)                        SEED="$2"; shift 2 ;;
            --on-missing)                  ON_MISSING="$2"; shift 2 ;;
            --push)                        PUSH=1; shift ;;
            --with-debate)                 WITH_DEBATE=1; shift ;;
            --skip-sound)                  SKIP_SOUND=1; shift ;;
            --skip-han-viet)               SKIP_HAN_VIET=1; shift ;;
            --skip-phuong-ngu)             SKIP_PHUONG_NGU=1; shift ;;
            --skip-tu-muon)                SKIP_TU_MUON=1; shift ;;
            --skip-tu-lay)                 SKIP_TU_LAY=1; shift ;;
            --skip-finalize)               SKIP_FINALIZE=1; shift ;;
            --skip-push)                   SKIP_PUSH=1; shift ;;
            --dry-run)                     DRY_RUN=1; shift ;;
            -h|--help)                     usage; exit 0 ;;
            *)                             EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
}

# parse + resolve đúng 1 lần; các hàm cmd_* gọi hàm này nên `all` không phải parse lại.
_cmd_prelude() {
    [[ "${_INITIALIZED}" == "1" ]] && return 0
    parse_common "$@"
    [[ "${LOCATION}" == "drive" || "${LOCATION}" == "local" ]] || die "--location phải là drive hoặc local."
    resolve_paths
    _INITIALIZED=1
}

# Gán KG_ARGS = (--knowledge-json <file>) nếu file tồn tại, ngược lại rỗng + cảnh báo.
KG_ARGS=()
set_kg_args() {
    KG_ARGS=()
    local f="$1"
    if [[ -f "$f" ]]; then
        KG_ARGS=(--knowledge-json "$f")
    else
        warn "chưa có $f -- chạy KHÔNG knowledge graph (Gemini tự quyết định)."
    fi
}

# finalize pre-final.jsonl -> pre-final_final.jsonl
finalize_file() {
    local pre="$1"
    [[ -f "$pre" ]] || { warn "bỏ qua finalize (không có file): $pre"; return 0; }
    local final="${pre%.jsonl}_final.jsonl"
    py "${HIEN_TUONG_SRC_DIR}/finalize_qa.py" --pre-final "$pre" --final "$final" --on-missing "$ON_MISSING"
}

# =============================================================================================
# fetch-speech / fetch-speech-zip
# =============================================================================================
cmd_fetch_speech() {
    _cmd_prelude "$@"
    require_hf_read_token
    local audio_args=()
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        audio_args=("${EXTRA_ARGS[@]}")
    fi
    py "${TOOLS_DIR}/fetch_datasets.py" speech \
        --repo-id "$HF_SPEECH_REPO" \
        --local-dir "$VIETNAMESE_SPEECH_QA_LOCAL_DIR" \
        --token "$HF_TOKEN" \
        "${audio_args[@]+"${audio_args[@]}"}"
}

cmd_fetch_speech_zip() {
    _cmd_prelude "$@"
    require_hf_read_token
    py "${TOOLS_DIR}/fetch_datasets.py" speech-zip \
        --repo-id "$HF_SPEECH_REPO" \
        --local-dir "$VIETNAMESE_SPEECH_QA_LOCAL_DIR" \
        --token "$HF_TOKEN"
}

# =============================================================================================
# sound
# =============================================================================================
cmd_sound() {
    _cmd_prelude "$@"
    require_dir "$SOUND_SRC_DIR" "sound src"
    local sa
    sa="$(gemini_service_account_arg)"

    py "${SOUND_SRC_DIR}/clotho_aqa_pipeline.py" build-manifest --output-dir "$CLOTHO_AQA_DIR"
    py "${SOUND_SRC_DIR}/clotho_aqa_pipeline.py" filter-vn-relevance \
        --service-account-json "$sa" \
        --input "$CLOTHO_AQA_NON_YESNO_MANIFEST" \
        --cache "$CLOTHO_AQA_VN_FILTER_CACHE" \
        --output "$CLOTHO_AQA_VN_MANIFEST"
    py "${SOUND_SRC_DIR}/clotho_aqa_pipeline.py" translate \
        --service-account-json "$sa" \
        --input "$CLOTHO_AQA_VN_MANIFEST" \
        --output "$CLOTHO_AQA_VI_QA_OUTPUT"
    py "${SOUND_SRC_DIR}/sound_dataset_merge.py" \
        --mmau-manifest "$MMAU_TEST_MINI_PATH" \
        --mmau-audio-dir "$MMAU_TEST_MINI_AUDIO_DIR" \
        --clotho-manifest "$CLOTHO_AQA_VI_QA_OUTPUT" \
        --clotho-audio-dir "${CLOTHO_AQA_DIR}/audio" \
        --sound-out-dir "$SOUND_OUT_DIR" \
        --output "$TEST_SOUND_MANIFEST_PATH"

    if [[ "${PUSH}" == "1" ]]; then
        require_hf_write_token
        local zip_base="${SOUND_ZIP_PATH%.zip}"
        py -c 'import shutil,sys; shutil.make_archive(sys.argv[1], "zip", root_dir=sys.argv[2])' "$zip_base" "$SOUND_OUT_DIR"
        py "$HF_PR_PUSH" \
            --token "$HF_WRITE_TOKEN" \
            --repo-id "$HF_SOUND_REPO" \
            --file "${TEST_SOUND_MANIFEST_PATH}=test_sound.jsonl" \
            --file "${SOUND_ZIP_PATH}=sound.zip" \
            --commit-message "Add merged MMAU+ClothoAQA sound QA (test_sound.jsonl + sound.zip)"
    fi
}

# =============================================================================================
# debate (knowledge graph)
# =============================================================================================
cmd_debate() {
    _cmd_prelude "$@"
    require_var "TASK" "$TASK"

    local suffix=""
    [[ -n "$VARIANT" ]] && suffix="_${VARIANT}"
    local seed_json="${KNOWLEDGE_DIR}/debate_seed_${TASK}${suffix}.json"
    local kg_json="${KNOWLEDGE_DIR}/knowledge_${TASK}${suffix}.json"

    local seed_args=(--task "$TASK" --location "$LOCATION" --out-json "$seed_json")
    [[ -n "$VARIANT" ]] && seed_args+=(--variant "$VARIANT")
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && seed_args+=("${EXTRA_ARGS[@]}")
    py "${HIEN_TUONG_SRC_DIR}/build_debate_seed.py" "${seed_args[@]}"

    local relay_args=(run --task "$TASK" --location "$LOCATION" --debate-mode "$DEBATE_MODE"
        --seed-json "$seed_json" --knowledge-output "$kg_json")
    [[ -n "$VARIANT" ]] && relay_args+=(--variant "$VARIANT")

    local rounds="$MAX_ROUNDS"
    if [[ -z "$rounds" ]]; then
        [[ "$TASK" == "code_switching" ]] && rounds=10 || rounds=5
    fi
    relay_args+=(--max-rounds "$rounds")

    # Code-switching có candidate list lớn -> chia batch; các task nhỏ để mặc định 1 batch.
    if [[ "$TASK" == "code_switching" ]]; then
        relay_args+=(--batch-size "${BATCH_SIZE:-24}" --max-workers "${MAX_WORKERS:-8}")
    else
        [[ -n "$BATCH_SIZE" ]] && relay_args+=(--batch-size "$BATCH_SIZE")
        [[ -n "$MAX_WORKERS" ]] && relay_args+=(--max-workers "$MAX_WORKERS")
    fi
    [[ -n "$MAX_RETRIES" ]] && relay_args+=(--max-retries "$MAX_RETRIES")
    [[ -n "$MODEL" ]] && relay_args+=(--gemini-model "$MODEL")

    py "${HIEN_TUONG_SRC_DIR}/auto_model_relay.py" "${relay_args[@]}"
}

# =============================================================================================
# han-viet
# =============================================================================================
HAN_VIET_FIELD_MAP='{"meaning": "Nghĩa", "category_hint": "Phạm Trù Gợi Ý", "historical_fact": "Sự Kiện Lịch Sử"}'
HAN_VIET_METADATA_COLS='{"tu": "Từ Hán Việt", "meaning": "Nghĩa", "category_hint": "Phạm Trù Gợi Ý", "historical_fact": "Sự Kiện Lịch Sử"}'

cmd_han_viet() {
    _cmd_prelude "$@"
    local kg="${KNOWLEDGE_DIR}/knowledge_han_viet.json"
    local sa
    sa="$(gemini_service_account_arg)"
    local knowledge_args=()
    if [[ -f "$kg" ]]; then knowledge_args=(--knowledge-json "$kg"); else warn "chưa có $kg -- classify/generate chạy không KG."; fi

    # Bước 0: seed CSV + báo cáo độ phủ
    py "${HAN_VIET_SRC_DIR}/han_viet_seed_csv.py" --input "$HAN_VIET_INPUT_PATH" --out-csv "$HAN_VIET_SEED_CSV"
    py "${HIEN_TUONG_SRC_DIR}/hien_tuong_filter_pipeline.py" word-coverage-report \
        --csv "$HAN_VIET_SEED_CSV" --word-col "Từ Hán Việt" \
        --full-transcripts-json "$FULL_TRANSCRIPTS_JSON_PATH" \
        --out-json "$HAN_VIET_COVERAGE_REPORT"

    # Bước 1: phân loại mức độ
    py "${HAN_VIET_SRC_DIR}/han_viet_pipeline.py" classify-levels \
        --service-account-json "$sa" \
        --input "$HAN_VIET_INPUT_PATH" \
        --output "$HAN_VIET_DIFFICULTY_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"

    # Bước 2: xuất CSV cuối từ knowledge graph + tìm sample cho từ MỚI + ghép record
    if [[ -f "$kg" ]]; then
        py "${HIEN_TUONG_SRC_DIR}/apply_knowledge_graph.py" \
            --knowledge-json "$kg" \
            --base-csv "$HAN_VIET_SEED_CSV" --word-col "Từ Hán Việt" \
            --field-map "$HAN_VIET_FIELD_MAP" \
            --out-csv "$HAN_VIET_FINAL_CSV"
        py "${HIEN_TUONG_SRC_DIR}/hien_tuong_filter_pipeline.py" build-samples \
            --csv "$HAN_VIET_FINAL_CSV" --word-col "Từ Hán Việt" \
            --full-transcripts-json "$RELEASE_HF_TRANSCRIPTS_JSON_PATH" \
            --metadata-cols "$HAN_VIET_METADATA_COLS" \
            --list-key han_viet_xuat_hien --out-json "$HAN_VIET_NEW_WORD_SAMPLES"
        py "${HAN_VIET_SRC_DIR}/han_viet_pipeline.py" build-new-word-records \
            --samples "$HAN_VIET_NEW_WORD_SAMPLES" --list-key han_viet_xuat_hien \
            --append-to "$HAN_VIET_DIFFICULTY_OUTPUT"
    else
        warn "Bỏ qua bước apply_knowledge_graph/build-samples/build-new-word-records (không có $kg)."
    fi

    # Bước 3: sinh câu hỏi + join field gốc
    py "${HAN_VIET_SRC_DIR}/han_viet_pipeline.py" generate-questions \
        --service-account-json "$sa" \
        --input "$HAN_VIET_DIFFICULTY_OUTPUT" \
        --output "$HAN_VIET_MULTIHOP_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"
    require_file "$TEST_SPEECH_JSONL" "test_speech.jsonl (chạy fetch-speech trước)"
    py "${HAN_VIET_SRC_DIR}/han_viet_pipeline.py" fill-fields \
        --multihop "$HAN_VIET_MULTIHOP_OUTPUT" \
        --original "$TEST_SPEECH_JSONL"

    finalize_file "$HAN_VIET_MULTIHOP_OUTPUT"
}

# =============================================================================================
# phuong-ngu
# =============================================================================================
cmd_phuong_ngu() {
    _cmd_prelude "$@"
    local kg="${KNOWLEDGE_DIR}/knowledge_phuong_ngu.json"
    local sa
    sa="$(gemini_service_account_arg)"
    local knowledge_args=()
    if [[ -f "$kg" ]]; then knowledge_args=(--knowledge-json "$kg"); else warn "chưa có $kg -- chạy không KG."; fi

    py "${PHUONG_NGU_SRC_DIR}/phuong_ngu_pipeline.py" classify-region \
        --service-account-json "$sa" \
        --input "$PHUONG_NGU_INPUT_PATH" \
        --output "$PHUONG_NGU_REGION_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"
    py "${PHUONG_NGU_SRC_DIR}/phuong_ngu_pipeline.py" generate-questions \
        --service-account-json "$sa" \
        --input "$PHUONG_NGU_REGION_OUTPUT" \
        --output "$PHUONG_NGU_QA_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"

    finalize_file "$PHUONG_NGU_QA_OUTPUT"
}

# =============================================================================================
# tu-muon
# =============================================================================================
cmd_tu_muon() {
    _cmd_prelude "$@"
    local kg="${KNOWLEDGE_DIR}/knowledge_tu_muon.json"
    local sa
    sa="$(gemini_service_account_arg)"
    local knowledge_args=()
    if [[ -f "$kg" ]]; then knowledge_args=(--knowledge-json "$kg"); else warn "chưa có $kg -- chạy không KG."; fi

    py "${TU_MUON_SRC_DIR}/tu_muon_pipeline.py" classify-levels \
        --service-account-json "$sa" \
        --samples "$TU_MUON_SAMPLES_PATH" \
        --csv "$TU_MUON_CSV_PATH" \
        --output "$TU_MUON_DIFFICULTY_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"
    py "${TU_MUON_SRC_DIR}/tu_muon_pipeline.py" generate-questions \
        --service-account-json "$sa" \
        --input "$TU_MUON_DIFFICULTY_OUTPUT" \
        --csv "$TU_MUON_CSV_PATH" \
        --output "$TU_MUON_MULTIHOP_OUTPUT" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"

    finalize_file "$TU_MUON_MULTIHOP_OUTPUT"
}

# =============================================================================================
# tu-lay
# =============================================================================================
TU_LAY_CLOZE_SAMPLES=""
cmd_tu_lay_variant() {
    local variant="$1"
    tu_lay_variant_paths "$variant"
    local kg="${KNOWLEDGE_DIR}/knowledge_tu_lay_${variant}.json"
    local sa
    sa="$(gemini_service_account_arg)"
    local knowledge_args=()
    if [[ -f "$kg" ]]; then knowledge_args=(--knowledge-json "$kg"); else warn "chưa có $kg -- chạy không KG."; fi

    py "${TU_LAY_SRC_DIR}/tu_lay_pipeline.py" classify-levels --variant "$variant" \
        --service-account-json "$sa" \
        --samples "$TU_LAY_VARIANT_SAMPLES" \
        --csv "$TU_LAY_VARIANT_CSV" \
        --output "$TU_LAY_VARIANT_DIFFICULTY" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"
    py "${TU_LAY_SRC_DIR}/tu_lay_pipeline.py" generate-questions --variant "$variant" \
        --service-account-json "$sa" \
        --input "$TU_LAY_VARIANT_DIFFICULTY" \
        --csv "$TU_LAY_VARIANT_CSV" \
        --output "$TU_LAY_VARIANT_MULTIHOP" \
        "${knowledge_args[@]+"${knowledge_args[@]}"}"

    finalize_file "$TU_LAY_VARIANT_MULTIHOP"
}

cmd_tu_lay_cloze() {
    # Chỉ variant toan_bo; cần cột "Từ gốc (cơ sở)" (do apply_knowledge_graph.py sinh ra).
    tu_lay_variant_paths "toan_bo"
    TU_LAY_CLOZE_SAMPLES="${TU_LAY_FINAL_DIR}/asr_samples_with_tu_lay_toan_bo_cloze.json"
    local cloze_output="${TU_LAY_FINAL_DIR}/tu_lay_toan_bo_cloze_qa.jsonl"

    py "${HIEN_TUONG_SRC_DIR}/hien_tuong_filter_pipeline.py" build-cloze-samples \
        --csv "$TU_LAY_VARIANT_CSV" --word-col "Từ láy" --base-word-col "Từ gốc (cơ sở)" \
        --full-transcripts-json "$RELEASE_HF_TRANSCRIPTS_JSON_PATH" \
        --list-key tu_lay_cloze_candidate --out-json "$TU_LAY_CLOZE_SAMPLES"
    py "${TU_LAY_SRC_DIR}/tu_lay_pipeline.py" generate-cloze-questions \
        --samples "$TU_LAY_CLOZE_SAMPLES" \
        --csv "$TU_LAY_VARIANT_CSV" \
        --output "$cloze_output"

    finalize_file "$cloze_output"
}

cmd_tu_lay() {
    _cmd_prelude "$@"
    local v="${VARIANT:-all}"
    case "$v" in
        all)
            cmd_tu_lay_variant toan_bo
            cmd_tu_lay_cloze
            cmd_tu_lay_variant van
            cmd_tu_lay_variant chung
            ;;
        toan_bo)
            cmd_tu_lay_variant toan_bo
            cmd_tu_lay_cloze
            ;;
        van|chung)
            cmd_tu_lay_variant "$v"
            ;;
        *) die "--variant phải là toan_bo|van|chung|all (nhận: $v)." ;;
    esac
}

# =============================================================================================
# finalize / push-hf
# =============================================================================================
cmd_finalize() {
    _cmd_prelude "$@"
    finalize_file "$HAN_VIET_MULTIHOP_OUTPUT"
    finalize_file "$PHUONG_NGU_QA_OUTPUT"
    finalize_file "$TU_MUON_MULTIHOP_OUTPUT"
    tu_lay_variant_paths toan_bo
    finalize_file "$TU_LAY_VARIANT_MULTIHOP"
    finalize_file "${TU_LAY_FINAL_DIR}/tu_lay_toan_bo_cloze_qa.jsonl"
    tu_lay_variant_paths van
    finalize_file "$TU_LAY_VARIANT_MULTIHOP"
    tu_lay_variant_paths chung
    finalize_file "$TU_LAY_VARIANT_MULTIHOP"
}

cmd_push_hf() {
    _cmd_prelude "$@"
    require_hf_write_token
    local target="${TARGET:-all}"

    if [[ "$target" == "speech" || "$target" == "all" ]]; then
        local prefix="speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
        py "$HF_PR_PUSH" \
            --token "$HF_WRITE_TOKEN" \
            --repo-id "$HF_SPEECH_REPO" \
            --file "${HAN_VIET_FINAL_QA}=${prefix}/han_viet/han_viet_multihop_qa_final.jsonl" \
            --file "${PHUONG_NGU_FINAL_QA}=${prefix}/phuong_ngu/phuong_ngu_region_qa_final.jsonl" \
            --file "${TU_MUON_FINAL_QA}=${prefix}/tu_muon/tu_muon_multihop_qa_final.jsonl" \
            --file "${TU_LAY_FINAL_DIR}/tu_lay_toan_bo_multihop_qa_final.jsonl=${prefix}/tu_lay/tu_lay_toan_bo_multihop_qa_final.jsonl" \
            --file "${TU_LAY_FINAL_DIR}/tu_lay_toan_bo_cloze_qa_final.jsonl=${prefix}/tu_lay/tu_lay_toan_bo_cloze_qa_final.jsonl" \
            --file "${TU_LAY_FINAL_DIR}/tu_lay_van_multihop_qa_final.jsonl=${prefix}/tu_lay/tu_lay_van_multihop_qa_final.jsonl" \
            --file "${TU_LAY_FINAL_DIR}/tu_lay_chung_multihop_qa_final.jsonl=${prefix}/tu_lay/tu_lay_chung_multihop_qa_final.jsonl" \
            --commit-message "Add Hán Việt/Phương ngữ/Từ mượn/Từ láy multi-hop reasoning QA (Hiện tượng đặc biệt tiếng Việt)"
    fi

    if [[ "$target" == "sound" || "$target" == "all" ]]; then
        py "$HF_PR_PUSH" \
            --token "$HF_WRITE_TOKEN" \
            --repo-id "$HF_SOUND_REPO" \
            --file "${TEST_SOUND_MANIFEST_PATH}=test_sound.jsonl" \
            --file "${SOUND_ZIP_PATH}=sound.zip" \
            --commit-message "Add merged MMAU+ClothoAQA sound QA (test_sound.jsonl + sound.zip)"
    fi
}

# =============================================================================================
# code-switching
# =============================================================================================
cmd_code_switching_mmsu() {
    _cmd_prelude "$@"
    local sa_args=()
    [[ -f "$SERVICE_ACCOUNT_JSON" ]] && sa_args=(--service-account-json "$SERVICE_ACCOUNT_JSON")
    py "${CODE_SWITCHING_SRC_DIR}/code_switching_pipeline.py" build-manifest \
        --location "$LOCATION" \
        "${sa_args[@]+"${sa_args[@]}"}"
}

cmd_code_switching() {
    _cmd_prelude "$@"
    local cs_pipeline="${CODE_SWITCHING_SRC_DIR}/code_switching_pipeline.py"
    local cs_qa="${CODE_SWITCHING_SRC_DIR}/code_switching_qa_pipeline.py"

    # 1. Chuẩn bị dữ liệu thật (GigaSpeech2-vi scan từ điển + hợp nhất ViMed)
    py "$cs_pipeline" scan-dictionary --location "$LOCATION"
    py "$cs_pipeline" merge-datasets --location "$LOCATION"

    # 2. Tri thức nền (debate) -- build_debate_seed mặc định theo task code_switching
    py "${HIEN_TUONG_SRC_DIR}/build_debate_seed.py" --task code_switching --location "$LOCATION"
    local rounds="${MAX_ROUNDS:-10}"
    py "${HIEN_TUONG_SRC_DIR}/auto_model_relay.py" run --task code_switching \
        --location "$LOCATION" --debate-mode "$DEBATE_MODE" \
        --max-rounds "$rounds" \
        --batch-size "${BATCH_SIZE:-24}" --max-workers "${MAX_WORKERS:-8}"

    # 3. Sinh câu hỏi + lọc 2 lượt
    py "$cs_qa" classify-cs --location "$LOCATION"
    py "$cs_qa" generate-questions --location "$LOCATION"
    py "$cs_qa" filter-questions --provider gemini --location "$LOCATION"
    py "$cs_qa" filter-questions --provider openai --location "$LOCATION"
}

# =============================================================================================
# inspect
# =============================================================================================
cmd_inspect() {
    _cmd_prelude "$@"
    [[ ${#EXTRA_ARGS[@]} -eq 0 ]] && die "inspect cần subcommand: tree|stats|sample|head|knowledge-status"
    if [[ "${EXTRA_ARGS[0]}" == "knowledge-status" && ${#EXTRA_ARGS[@]} -eq 1 ]]; then
        EXTRA_ARGS+=("$KNOWLEDGE_DIR")
    fi
    py "${TOOLS_DIR}/inspect_qa.py" "${EXTRA_ARGS[@]}"
}

# =============================================================================================
# local-preprocess (chạy ở máy local, cần stuff/benchmark_qa/)
# =============================================================================================
cmd_local_preprocess() {
    _cmd_prelude "$@"
    require_var "TASK" "$TASK"
    local base="${REPO_DIR}/stuff/benchmark_qa"
    local pipeline="${HIEN_TUONG_SRC_DIR}/hien_tuong_filter_pipeline.py"
    local vlsp_glob="asr_datasets/vlsp/*.json"
    local ledger="asr_datasets/asr_source_ledger.json"

    case "$TASK" in
        tu-muon)
            py "$pipeline" filter-csv \
                --src-csv "$base/tu_muon_tieng_viet_viet_hoa.csv" --word-col "Từ Tiếng Việt (Việt Hóa)" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --out-csv "$base/tu_muon_tieng_viet_viet_hoa_final.csv"
            py "$pipeline" build-samples \
                --csv "$base/tu_muon_tieng_viet_viet_hoa_final.csv" --word-col "Từ Tiếng Việt (Việt Hóa)" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --list-key tu_muon_xuat_hien \
                --metadata-cols '{"tu": "Từ Tiếng Việt (Việt Hóa)", "tu_goc": "Từ Gốc", "ngon_ngu_nguon_goc": "Ngôn Ngữ / Nguồn Gốc", "nhom_linh_vuc": "Nhóm Lĩnh Vực", "y_nghia_ghi_chu": "Ý Nghĩa / Ghi Chú"}' \
                --out-json "$base/asr_datasets/asr_samples_with_tu_muon.json"
            ;;
        tu-lay)
            py "$pipeline" filter-csv \
                --src-csv "$base/tu_lay_toan_bo_va_van.csv" --word-col "Từ láy" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --out-csv "$base/tu_lay_toan_bo_va_van_final.csv"
            py "$pipeline" build-samples \
                --csv "$base/tu_lay_toan_bo_va_van_final.csv" --word-col "Từ láy" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --list-key tu_lay_xuat_hien \
                --metadata-cols '{"tu": "Từ láy", "phan_loai": "Phân loại", "y_nghia": "Ý nghĩa", "sac_thai_bieu_dat": "Sắc thái biểu đạt"}' \
                --out-json "$base/asr_datasets/asr_samples_with_tu_lay_toan_bo_va_van.json"
            py "$pipeline" split-by-prefix \
                --csv "$base/tu_lay_toan_bo_va_van_final.csv" --column "Phân loại" --prefix "Láy toàn bộ" \
                --out-csv-matched "$base/tu_lay_toan_bo_final.csv" --out-csv-rest "$base/tu_lay_van_final.csv" \
                --samples "$base/asr_datasets/asr_samples_with_tu_lay_toan_bo_va_van.json" --word-col "Từ láy" \
                --list-key tu_lay_xuat_hien --metadata-word-key tu \
                --out-samples-matched "$base/asr_datasets/asr_samples_with_tu_lay_toan_bo.json" \
                --out-samples-rest "$base/asr_datasets/asr_samples_with_tu_lay_van.json"
            py "$pipeline" filter-csv \
                --src-csv "$base/tu_lay_tieng_viet.csv" --word-col "Từ láy" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --out-csv "$base/tu_lay_tieng_viet_final.csv"
            py "$pipeline" build-samples \
                --csv "$base/tu_lay_tieng_viet_final.csv" --word-col "Từ láy" \
                --vlsp-glob "$base/$vlsp_glob" --ledger "$base/$ledger" \
                --list-key tu_lay_xuat_hien \
                --metadata-cols '{"tu": "Từ láy", "loai_tu_lay": "Loại từ láy", "tu_loai": "Từ loại", "y_nghia": "Ý nghĩa", "sac_thai_bieu_dat": "Sắc thái biểu đạt"}' \
                --out-json "$base/asr_datasets/asr_samples_with_tu_lay.json"
            py "$pipeline" dedupe-samples \
                --samples "$base/asr_datasets/asr_samples_with_tu_lay.json" --list-key tu_lay_xuat_hien --word-key tu \
                --against "$base/asr_datasets/asr_samples_with_tu_lay_toan_bo.json" \
                --against "$base/asr_datasets/asr_samples_with_tu_lay_van.json" \
                --output "$base/asr_datasets/asr_samples_with_tu_lay.json"
            ;;
        *) die "--task phải là tu-muon hoặc tu-lay (nhận: $TASK)." ;;
    esac
}

# =============================================================================================
# all
# =============================================================================================
cmd_all() {
    _cmd_prelude "$@"
    log "=== all: bắt đầu (location=$LOCATION, dry_run=$DRY_RUN) ==="
    if [[ "$SKIP_SOUND" == "0" ]]; then cmd_sound; else log "bỏ qua sound"; fi
    if [[ "$WITH_DEBATE" == "1" ]]; then log "(--with-debate: chạy debate riêng cho từng task nếu cần)"; fi
    if [[ "$SKIP_HAN_VIET" == "0" ]]; then cmd_han_viet; else log "bỏ qua han-viet"; fi
    if [[ "$SKIP_PHUONG_NGU" == "0" ]]; then cmd_phuong_ngu; else log "bỏ qua phuong-ngu"; fi
    if [[ "$SKIP_TU_MUON" == "0" ]]; then cmd_tu_muon; else log "bỏ qua tu-muon"; fi
    if [[ "$SKIP_TU_LAY" == "0" ]]; then cmd_tu_lay; else log "bỏ qua tu-lay"; fi
    if [[ "$SKIP_FINALIZE" == "0" ]]; then cmd_finalize; else log "bỏ qua finalize"; fi
    if [[ "$PUSH" == "1" && "$SKIP_PUSH" == "0" ]]; then
        cmd_push_hf
    else
        log "bỏ qua push-hf (thêm --push để bật)"
    fi
    log "=== all: hoàn tất ==="
}

# =============================================================================================
# main
# =============================================================================================
main() {
    [[ $# -eq 0 ]] && { usage; exit 1; }
    local command="$1"
    shift
    case "$command" in
        fetch-speech)          cmd_fetch_speech "$@" ;;
        fetch-speech-zip)      cmd_fetch_speech_zip "$@" ;;
        sound)                 cmd_sound "$@" ;;
        debate)                cmd_debate "$@" ;;
        han-viet)              cmd_han_viet "$@" ;;
        phuong-ngu)            cmd_phuong_ngu "$@" ;;
        tu-muon)               cmd_tu_muon "$@" ;;
        tu-lay)                cmd_tu_lay "$@" ;;
        finalize)              cmd_finalize "$@" ;;
        push-hf)               cmd_push_hf "$@" ;;
        code-switching)        cmd_code_switching "$@" ;;
        code-switching-mmsu)   cmd_code_switching_mmsu "$@" ;;
        inspect)               cmd_inspect "$@" ;;
        local-preprocess)      cmd_local_preprocess "$@" ;;
        all)                   cmd_all "$@" ;;
        -h|--help|help)        usage ;;
        *)                     usage; die "Subcommand không hợp lệ: $command" ;;
    esac
}

main "$@"
