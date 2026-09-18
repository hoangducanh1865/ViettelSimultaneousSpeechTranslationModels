# ViettelSimultaneousSpeechTranslationModels

Repo xây **test set cho dịch/ASR tiếng Việt** và **benchmark QA** (sound / speech / code-switching),
cùng các công cụ phụ trợ (ASR ensemble + ROVER, web gán nhãn, chấm điểm dịch GEMBA, voice cloning).

Toàn bộ pipeline Benchmark QA trước đây nằm trong
`noteboooks/benchmark_qa/benchmark_qa_constructer copy 3.ipynb` đã được **gộp vào script** trong
`src/`; notebook giờ chỉ còn 3 cell (mount Drive → clone repo → chạy bash).

---

## 1. Cấu trúc repo

```
.
├── README.md                          # tài liệu này
├── requirements.txt
├── noteboooks/                        # notebook khám phá (BỊ GITIGNORE, không push)
└── src/
    ├── test_set/
    │   ├── datasets_qa/               # xây benchmark QA
    │   │   ├── translate_datasets/    # translate_dataset.py, hf_pr_push.py, generate_report.py
    │   │   └── benchmark_qa/
    │   │       ├── run/               # <-- entrypoint bash
    │   │       │   ├── benchmark_qa.sh
    │   │       │   ├── _lib.sh
    │   │       │   └── config.env.example
    │   │       ├── tools/             # inspect_qa.py, fetch_datasets.py
    │   │       ├── sound/             # ClothoAQA pipeline + merge với MMAU
    │   │       └── speech/
    │   │           ├── cac_hien_tuong_dac_biet_trong_tieng_viet/   # 4 task + script dùng chung
    │   │           └── trich_xuat_thong_tin/code_switching/        # Code-switching
    │   ├── public/                    # ASR clients + ROVER + manifest/versioning + HF publish
    │   ├── vietnamese_people_speak_english/   # crawl YouTube -> test set
    │   └── synthesize/                # voice cloning
    ├── labeling_app/                  # FastAPI + Next.js app gán nhãn
    └── gemba_vendor/                  # GEMBA (Microsoft) đã vendor -- chấm điểm dịch
```

---

## 2. Benchmark QA — chạy bằng bash

### 2.1 Cấu hình (đọc từ `.env`, KHÔNG truyền token qua cờ)

`run/benchmark_qa.sh` tự source theo thứ tự: `.env` ở gốc repo → `run/.env` (file sau ghi đè).
Copy mẫu rồi điền:

```bash
cp src/test_set/datasets_qa/benchmark_qa/run/config.env.example \
   src/test_set/datasets_qa/benchmark_qa/run/.env
# điền HF_TOKEN và HF_WRITE_TOKEN (bắt buộc), các biến khác tùy chọn
```

| Biến | Bắt buộc | Ý nghĩa |
|---|---|---|
| `HF_TOKEN` | khi tải dữ liệu HF | Token đọc HuggingFace |
| `HF_WRITE_TOKEN` | khi push | Token tạo PR lên HuggingFace |
| `LOCATION` | không (mặc định `drive`) | `drive` (Colab) hoặc `local` (máy bạn) |
| `REPO_DIR` | không | Gốc repo (tự suy từ vị trí script) |
| `QA_DATASETS_DIR` | không | Gốc dữ liệu QA: Drive, hoặc `<repo>/data` khi `LOCATION=local` |
| `ENV_FILE` | không | File credential: `<drive>/voice_agent_for_edge_device/.env` (drive) / `<repo>/.env` (local) |
| `SERVICE_ACCOUNT_JSON` | không | Service account Gemini (Vertex AI); bỏ trống thì dùng `ENV_FILE` proxy |
| `KNOWLEDGE_DIR` | không | Nơi chứa `knowledge_<task>.json` |
| `FULL_TRANSCRIPTS_JSON_PATH` | không | Corpus text rộng (bằng chứng coverage) |
| `RELEASE_HF_TRANSCRIPTS_JSON_PATH` | không | Corpus **có audio** (tạo sample mới) |
| `VIETNAMESE_SPEECH_QA_LOCAL_DIR` | không | `local_dir` tải Vietnamese-Speech-QA |
| `HF_SPEECH_REPO` / `HF_SOUND_REPO` | không | Repo đích khi push |
| `DEBATE_MODE` | không (mặc định `api`) | `api` (2 API tự debate) hoặc `manual` (copy-paste) |
| `GEMINI_MODEL` / `OPENAI_MODEL` | không | Model mạnh cho debate + lọc (mặc định `gemini-3.1-pro-preview` / `cx/gpt-5.6-luna`) |

> Hai token **bắt buộc nằm trong `.env`**, không nhận qua `--hf-token/--hf-write-token`.

### 2.2 Notebook 3 cell

```python
# Cell 1 -- mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 -- clone/pull repo
# Cell 3 -- chạy
```

Cell 2:

```bash
!bash -c 'REPO_DIR=/content/ViettelSimultaneousSpeechTranslationModels; \
  if [ ! -f "$REPO_DIR/src/test_set/datasets_qa/translate_datasets/translate_dataset.py" ]; then \
    git clone --depth 1 https://github.com/hoangducanh1865/ViettelSimultaneousSpeechTranslationModels.git "$REPO_DIR"; \
  else git -C "$REPO_DIR" pull; fi'
```

Cell 3 (chạy cả pipeline):

```bash
!bash /content/ViettelSimultaneousSpeechTranslationModels/src/test_set/datasets_qa/benchmark_qa/run/benchmark_qa.sh all --location drive
```

### 2.3 Quickstart (máy local / terminal)

```bash
cd src/test_set/datasets_qa/benchmark_qa
bash run/benchmark_qa.sh --help

bash run/benchmark_qa.sh all --location drive
bash run/benchmark_qa.sh sound --location drive --push
bash run/benchmark_qa.sh tu-lay --variant toan_bo --location drive
bash run/benchmark_qa.sh debate --task code_switching --location local --debate-mode manual
bash run/benchmark_qa.sh inspect knowledge-status
```

### 2.4 Danh sách subcommand

> **Luồng chuẩn của mọi task**: `[debate --debate-mode manual|api]` → sinh câu hỏi →
> **lọc 2 API model mạnh** (`filter_qa_pipeline.py`: Gemini `gemini-3.1-pro-preview` rồi OpenAI
> `cx/gpt-5.6-luna`) → `finalize_qa.py` → `*_final.jsonl`. Chế độ `manual` và `api` chạy y hệt nhau
> ở mọi bước sau debate. Bỏ qua lọc bằng `--skip-filter`; bước debate chỉ chạy khi `--debate` và
> `knowledge_<task>.json` chưa có.

| Subcommand | Việc nó chạy |
|---|---|
| `fetch-speech` | Tải `MAPPING_REPORT.json` + `test_speech.jsonl` (thêm audio subset bằng `--sino-audio`/`--dialect-audio`) |
| `fetch-speech-zip` | Tải + giải nén `speech.zip` đã publish |
| `sound` | ClothoAQA: `build-manifest` → `filter-vn-relevance` → `translate` → `sound_dataset_merge` (thêm `--push` để zip + PR) |
| `debate` | `build_debate_seed` → `auto_model_relay` (cần `--task`, `--debate-mode manual|api`) |
| `han-viet` | `[debate]` → `han_viet_seed_csv` → `word-coverage-report` → `classify-levels` → `apply_knowledge_graph` → `build-samples` → `build-new-word-records` → `generate-questions` → `fill-fields` → **lọc 2 API** → `finalize` |
| `phuong-ngu` | `[debate]` → `classify-region` → `generate-questions` → **lọc 2 API** → `finalize` |
| `tu-muon` | `[debate]` → `classify-levels` → `generate-questions` → **lọc 2 API** → `finalize` |
| `tu-lay` | `[debate]` → 3 variant (`toan_bo`/`van`/`chung`) + cloze cho `toan_bo` → **lọc 2 API** → `finalize` |
| `finalize` | Chỉ chạy `finalize_qa.py` cho các file pre-final hiện có |
| `push-hf` | Đẩy file final lên HF (`--target speech|sound|all`) |
| `code-switching` | `scan-dictionary` → `merge-datasets` → debate → `classify-cs` → `generate-questions` → **lọc 2 API** → `finalize` |
| `code-switching-mmsu` | Nhánh MMSU: `build-manifest` 111 sample Code-switching |
| `inspect` | `tree` / `stats` / `sample` / `head` / `knowledge-status` |
| `local-preprocess` | Tiền xử lý local-only (`--task tu-muon|tu-lay`) |
| `all` | sound + 4 task + code-switching (có `--skip-*`) |

### 2.5 Cờ dùng chung

```
--location drive|local            --repo-dir DIR
--qa-datasets-dir DIR             --service-account-json FILE
--knowledge-dir DIR               --full-transcripts-json FILE
--release-hf-transcripts-json F   --speech-local-dir DIR
--python BIN                      --task NAME
--variant toan_bo|van|chung|all   --target speech|sound|all
--on-missing skip|raise           --debate
--debate-mode api|manual          (mặc định: api)
--gemini-model NAME               (mặc định: gemini-3.1-pro-preview)
--openai-model NAME               (mặc định: cx/gpt-5.6-luna)
--gemini-base-url URL --openai-base-url URL --openai-api-key-file FILE --env-file FILE
--model --batch-size --max-workers --max-retries --max-rounds --seed
--skip-filter
--push    --with-debate    --skip-sound --skip-han-viet --skip-phuong-ngu
--skip-tu-muon --skip-tu-lay --skip-code-switching --skip-push    --dry-run
```

### 2.5b Lệnh chạy luồng 2 API debate + lọc 2 API (model mạnh)

```bash
# Toàn bộ (4 task + code-switching), debate tự động 2 API
bash run/benchmark_qa.sh all --location drive --debate-mode api

# Toàn bộ, debate copy-paste thủ công -- các bước sau debate chạy Y HỆT chế độ api
bash run/benchmark_qa.sh all --location local --debate-mode manual

# 1 task kèm debate trước khi sinh (chỉ debate khi knowledge_<task>.json chưa có)
bash run/benchmark_qa.sh code-switching --location local --debate --debate-mode api
bash run/benchmark_qa.sh tu-lay --location drive --debate --debate-mode api
bash run/benchmark_qa.sh han-viet --location drive --debate --debate-mode api

# Chỉ đổi model mạnh dùng cho debate + 2 lượt lọc
bash run/benchmark_qa.sh all --location local --debate-mode api \
  --gemini-model gemini-3.1-pro-preview --openai-model cx/gpt-5.6-luna

# Bỏ bước lọc (chỉ finalize), hoặc đổi cấp độ lọc
bash run/benchmark_qa.sh tu-lay --debate --skip-filter
```

`filter_qa_pipeline.py` dùng **prompt lọc chi tiết theo từng task** (tiêu chí chung: tự nhiên /
đa dạng / độ khó thực chất / nhiễu hợp lý / không lộ đáp án / chống đoán mò) cộng tiêu chí đặc thù:
Phương ngữ (câu phải hỏi vùng/dân tộc), Từ láy (cloze Tone Harmony: đáp án là từ láy toàn bộ hợp lệ,
không bắc cầu âm vực; loại "nhẹ nhẹ" nếu phải "nhè nhẹ"), Hán Việt / Từ mượn (đúng level, fact
lịch sử thật), Code-switching (6 loại A–F). Filter gửi kèm context (`base_word`, `tone_register`,
`region_or_ethnic_group`, `level`, `historical_fact`) để chấm chính xác.

### 2.6 Kiểm tra nhanh dữ liệu

```bash
# Cây category / thống kê / sample
bash run/benchmark_qa.sh inspect tree  $QA_DATASETS_DIR/test_sound.jsonl
bash run/benchmark_qa.sh inspect stats $QA_DATASETS_DIR/Vietnamese-Speech-QA/test_speech.jsonl
bash run/benchmark_qa.sh inspect sample $QA_DATASETS_DIR/test_sound.jsonl --start 0 --n 3 --audio
bash run/benchmark_qa.sh inspect head  $QA_DATASETS_DIR/Vietnamese-Speech-QA/MAPPING_REPORT.json --n 100
bash run/benchmark_qa.sh inspect knowledge-status
```

---

## 3. Pipeline Benchmark QA — chi tiết kỹ thuật

Nguyên tắc gốc: **đáp án đúng luôn lấy verbatim từ dữ liệu thật** (transcript ASR + CSV do người
kiểm duyệt, hoặc kiến thức đã được 3-model debate đồng thuận) — LLM chỉ dùng để diễn đạt câu hỏi
và/hoặc sinh nhiễu. Mọi pipeline **resumable** (id ghép, cache) và các bước tốn kém dùng
batch song song + retry/backoff. `finalize_qa.py` là ngoại lệ (luôn ghi đè).

### 3.1 Tri thức nền (Knowledge Graph)

Mỗi hạng mục (Hán Việt, Phương ngữ, Từ mượn, Từ láy ×3 variant, Code-switching) **có thể** có 1 file
`knowledge_<task>[_<variant>].json` do `auto_model_relay.py` sinh (relay Gemini ↔ 1 model
OpenAI-compatible). Đây là bước **tùy chọn**: nếu `--knowledge-json` trỏ tới file chưa tồn tại,
pipeline tự bỏ qua và để Gemini tự quyết định (kém robust hơn nhưng vẫn chạy).

- `build_debate_seed.py` gộp bằng chứng thật (CSV + web candidates + coverage report + sample
  transcript chưa khớp) thành seed theo TỪ (4 task) hoặc theo SAMPLE (`code_switching`).
- `auto_model_relay.py run` chạy debate, dừng khi cả 2 model trả `CONSENSUS: FINAL` (hoặc tối đa
  `--max-rounds`), xuất knowledge graph.
- `apply_knowledge_graph.py` (Hán Việt/Từ mượn/Từ láy: `confirmed`/`corrected` giữ-sửa, `added`
  thêm dòng, `rejected` loại) — CSV sau đó dùng cho `build-samples`.

Deploy:
```bash
# Chạy riêng bước debate (manual hoặc api -- cùng xuất knowledge graph)
bash run/benchmark_qa.sh debate --task code_switching --location local --debate-mode manual
bash run/benchmark_qa.sh debate --task han_viet       --location drive --debate-mode api

# Hoặc gộp debate vào chính task (chỉ debate nếu knowledge_<task>.json chưa có)
bash run/benchmark_qa.sh han-viet --location drive --debate --debate-mode api
bash run/benchmark_qa.sh tu-lay   --location drive --debate --debate-mode api
```

### 3.2 Sound (`test_sound.jsonl`)

ClothoAQA test split → `filter-vn-relevance` (Gemini lọc ngữ cảnh Việt Nam, cache JSON,
**mặc định GIỮ khi API lỗi**) → `translate` (dịch + sinh 3 nhiễu) → `sound_dataset_merge` ghép với
MMAU test-mini. Id sinh **deterministic bằng md5** nên chạy lại không nhân bản audio. Push:
`test_sound.jsonl` + `sound.zip`.

### 3.3 4 task "Hiện tượng đặc biệt trong tiếng Việt"

Kiến trúc chung 3-hop: `classify-levels` (gán `max_level` ∈ {1,2,3} + fact) → `generate-questions`
(sinh đúng N câu, id `<id>__L<level>`, level 1/2 rule-based từ CSV, level 3 mới gọi Gemini).

| Task | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Hán Việt | nghĩa đen | phạm trù/lĩnh vực suy từ transcript | sự kiện lịch sử-văn hóa THẬT |
| Từ mượn | 1 khía cạnh (nguồn gốc/từ gốc/ý nghĩa) | "Nhóm Lĩnh Vực" trong CSV | bối cảnh du nhập lịch sử THẬT |
| Từ láy | khía cạnh cấu tạo (loại/vần) | "Sắc thái biểu đạt" trong CSV | ca dao/tục ngữ/tác phẩm văn học THẬT |
| Phương ngữ | `classify-region` xác định vùng/dân tộc | — | — |

- **Hán Việt** thêm: `han_viet_seed_csv.py` bootstrap CSV từ 95 record; `word-coverage-report` làm
  bằng chứng; nhánh **từ mới** (`apply_knowledge_graph` → `build-samples` trên corpus CÓ audio →
  `build-new-word-records`) lấp các từ `status: added`; cuối cùng `fill-fields` join metadata từ
  `test_speech.jsonl`.
- **Từ láy** có thêm **cloze/Tone Harmony** cho `toan_bo`: cho transcript chứa từ GỐC, hỏi từ láy
  nào phù hợp tăng/giảm sắc thái; nhiễu ưu tiên cùng âm vực (luật hòa âm vực).

### 3.4 Pre-final vs Final

Mỗi bước sinh ghi **pre-final** (nhiều field debug: `transcript`, `question_type`, `level`,
`historical_fact`/`cultural_fact`, `tone_register`...). `finalize_qa.py` cắt về đúng schema publish
(11 field: `id/audio_id/question/choices/answer/dataset/task/split/category/sub-category/difficulty`),
validate cứng: đúng 4 `choices` phân biệt, `answer` ∈ `choices`. **Chỉ file `*_final.jsonl` được
push lên HF**; pre-final ở lại để debug.

### 3.5 Code-switching (trích xuất thông tin)

- **Nhánh MMSU**: `ddwang2000/MMSU`, lọc `task_name == "code_switch_question_answering"`
  (111/5000 sample) → manifest.
- **Nhánh dữ liệu thật**: `scan-dictionary` khớp từ điển `cs_broad_new.txt` vào GigaSpeech2-vi
  (khớp token chính xác), `merge-datasets` hợp nhất với ViMed (đã có `cs_terms` xác thực) →
  `code_switching_qa.jsonl`. Sau đó debate (`auto_model_relay`, batch 24) → `classify-cs` →
  `generate-questions` (6 loại A–F) → **`filter-questions` 2 lượt** (`--gemini-model`
  `gemini-3.1-pro-preview` rồi `--openai-model` `cx/gpt-5.6-luna`) → `finalize_qa.py` →
  `code_switching_openai_kept_final.jsonl`.
- Chạy trọn luồng: `bash run/benchmark_qa.sh code-switching --location local --debate --debate-mode api`.

---

## 4. Thành phần khác trong repo

### 4.1 `src/test_set/public/` — ASR ensemble + ROVER
`lang_id` lọc ngôn ngữ → 4 voter (`chirp3`, `internal_whisper`, `phowhisper`, base) →
`rover.py` (NIST ROVER: word-transition network + weighted vote) → `llm_refine.py` (Gemini
clean/fuse/translate có guard diff) → `manifest_io.py`/`dataset_versioning.py` (manifest + version)
→ `hf_publish.py`. Chạy: `python3 src/test_set/public/rover.py --dataset-root DIR ...`.

### 4.2 `src/labeling_app/` — web gán nhãn
FastAPI + SQLAlchemy backend (`samples`, `/api/samples`, `/api/export/mt-check.csv`) + Next.js 14.
`prepare_data.py` chuyển manifest Colab → `seed_samples.json` + audio. Deploy Render + Vercel.

### 4.3 `src/gemba_vendor/` — GEMBA
Đánh giá chất lượng dịch bằng LLM (GEMBA-MQM/DA/SQM/ESA), vendor từ Microsoft, có patch local.
Credentials: `OPENAI_API_KEY` hoặc `OPENAI_AZURE_KEY` + `OPENAI_AZURE_ENDPOINT`.

### 4.4 `src/test_set/vietnamese_people_speak_english/`
`crawl.py` (yt-dlp) → `process_videos.py` (cắt segment + lang_id) → `translate_en_to_vi.py`
(ROVER + LLM dịch EN→VI).

### 4.5 `src/test_set/synthesize/voice_cloning.py`
Zero-shot voice cloning tiếng Việt (`hynt/ZipVoice-Vietnamese-2500h`, license NC).

---

## 5. Test E2E local (real API)

Bộ test `tests/benchmark_qa/` chạy **thật** qua bash dispatcher với dữ liệu nhỏ (2-3 mẫu) ghi trong
`data/_e2e/`, gồm cả tiền xử lý, cho cả 5 task (Code-switching có bước debate 2 API).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

RUN_REAL_API=1 PYTHONPATH=src python -m pytest tests/benchmark_qa -q
```

- Cần `GEMINI_API_KEY` + `OPENAI_API_KEY` (kèm `# base_url # model`) trong `<repo>/.env`, proxy
  phải đang chạy. Thiếu key → test tự `skip`.
- Nếu proxy dùng tên model khác mặc định, override:
  `E2E_GEMINI_MODEL=ag/gemini-3.x E2E_OPENAI_MODEL=cx/gpt-5.6-luna RUN_REAL_API=1 pytest ...`.

---

## 6. Ghi chú

- **Local dùng `data/` làm data root** (thay cho `stuff/` cũ): `--location local` trỏ mọi đường dẫn
  vào `<repo>/data/...`. `noteboooks/` vẫn bị gitignore.
- Không commit token. `.env` (gốc repo, local) và `.env` trên Drive đều bị ignore.
- `.env` dùng format `KEY="value" # base_url # model`; cả 4 task lẫn code-switching/filter đều đọc
  credential qua `auto_model_relay.resolve_role_client()` (Vertex service account nếu có, ngược lại
  dùng `.env` proxy).
- `PYTHONPATH` được `run/_lib.sh` tự trỏ vào `<repo>/src` để import
  `test_set.datasets_qa.translate_datasets.translate_dataset`.
