"""Drives a FULLY AUTOMATED relay between 2 API-backed models (Gemini + 1 OpenAI-compatible
model behind a custom base_url) to build ONE task's knowledge-graph JSON (see knowledge_graph.py
for the schema). Replaces manual_model_relay.py's copy-paste-into-web-chat mechanism: every turn
calls the model's API directly and reads its response programmatically -- no human paste step, no
--prompt-file/--response-file handoff, no input(). Because there is no interactive stdin need
anymore, this CAN run inside a Colab `!python ...` cell (unlike manual_model_relay.py before it).

Used for EVERY task -- the 4 pre-existing ones (Hán Việt/Phương ngữ/Từ mượn/Từ láy, whose
knowledge_*.json files already exist and do NOT need to be regenerated) and any new task (e.g.
Code-switching) going forward. Two seed shapes are supported, auto-detected from the seed JSON:
  - "candidate_words": flat per-word list (what build_debate_seed.py's CSV/word-list branch
    produces) -- the shape all 4 existing tasks use.
  - "candidate_units": per-sample groups, each carrying its own transcript + the FULL list of
    terms co-occurring in that sample (what build_debate_seed.py's --samples-jsonl branch
    produces) -- needed for Code-switching so multi-term relational context survives in 1 turn.

Scale: --batch-size (default None = 1 single batch containing every candidate, i.e. identical
single-shot behavior to the old 4-task flow) splits a big candidate/unit list into independent
batches, each run through its own full consensus loop (up to --max-rounds), batches executed
concurrently via --max-workers. Results are merged afterwards (merge_batch_knowledge) into 1
knowledge graph.

Stopping condition per batch: a round (one turn per model in --model-order) ends that batch's
relay the moment EVERY model's turn that round contains the literal marker "CONSENSUS: FINAL";
otherwise the batch continues until --max-rounds is exhausted, at which point the LATEST
successfully-parsed knowledge JSON for that batch is used regardless of consensus.

Usage:
    python auto_model_relay.py run --task tu_lay --variant toan_bo \\
        --seed-json debate_seed_tu_lay_toan_bo.json \\
        --state-output relay_state_tu_lay_toan_bo.json \\
        --knowledge-output knowledge_tu_lay_toan_bo.json \\
        --gemini-service-account-json gemini_service_account.json \\
        --openai-api-key-file openai_api_key.txt

    python auto_model_relay.py run --task code_switching \\
        --seed-json debate_seed_code_switching.json \\
        --state-output relay_state_code_switching.json \\
        --knowledge-output knowledge_code_switching.json \\
        --gemini-service-account-json gemini_service_account.json \\
        --openai-api-key-file openai_api_key.txt \\
        --max-rounds 10 --batch-size 8

    python auto_model_relay.py show --state relay_state_tu_lay_toan_bo.json

Chạy LOCAL (không cần service account/file key riêng, dùng chung 1 file .env dạng
KEY="value" # base_url # model -- xem load_env_file()):
    python auto_model_relay.py run --task code_switching \\
        --seed-json code_switching/debate_seed_code_switching.json \\
        --state-output code_switching/relay_state_code_switching.json \\
        --knowledge-output code_switching/knowledge_code_switching.json \\
        --env-file code_switching/.env \\
        --max-rounds 10 --batch-size 24 --max-workers 24
Không truyền --env-file thì mặc định tìm file ".env" cạnh --seed-json. Colab (service account +
file key) và local (.env) dùng CHUNG 1 lệnh/1 code -- resolve_clients() tự chọn nguồn credential
phù hợp dựa trên tham số nào được truyền, không cần biết trước đang chạy ở đâu.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from tqdm.auto import tqdm

import env_paths
from knowledge_graph import SCHEMA_VERSION

DEFAULT_MODEL_ORDER = ["gemini", "openai"]
DEFAULT_MANUAL_MODEL_ORDER = ["gemini", "chatgpt", "claude"]
DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OPENAI_MODEL = "cx/gpt-5.6-luna"
DEFAULT_OPENAI_BASE_URL = "https://r3wrrfi.abc-tunnel.us/v1"

_CONSENSUS_MARKER_RE = re.compile(r"CONSENSUS:\s*(FINAL|CONTINUE)", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# response quá dài đôi khi bị model BỎ QUÊN fence đóng "```" cuối (không phải bị cắt cụt bởi
# max_output_tokens -- response vẫn ngắn hơn nhiều so với giới hạn) -- 2 regex này chỉ bóc fence
# MỞ (có hoặc không có "json" theo sau) để dùng làm fallback khi _JSON_FENCE_RE (đòi cả 2 fence)
# không match được gì.
_LEADING_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_TRAILING_FENCE_RE = re.compile(r"```\s*$")


@dataclass
class Turn:
    round_index: int
    turn_index: int
    model: str
    prompt: str
    response: str
    parsed_knowledge: Optional[dict]
    consensus_vote: Optional[str]  # "FINAL" | "CONTINUE" | None (marker missing/call failed)


@dataclass
class RelayState:
    task: str
    variant: Optional[str]
    model_order: list[str]
    max_rounds: int
    turns: list[Turn] = field(default_factory=list)
    latest_knowledge: Optional[dict] = None
    stopped_reason: Optional[str] = None  # "consensus" | "max_rounds" | None (still running)


def _loads_lenient(text: str) -> Optional[dict]:
    """json.loads() nghiêm ngặt fail ngay khi có bất kỳ ký tự THỪA nào sau JSON hợp lệ (ví dụ
    model quên bọc fence nên "CONSENSUS: FINAL" bị dính liền sau dấu "}" cuối -> lỗi "Extra
    data"). json.JSONDecoder().raw_decode() chỉ parse ĐÚNG value JSON đầu tiên và bỏ qua phần
    thừa phía sau -- khoan dung hơn nhiều với các lỗi định dạng nhỏ của model mà vẫn an toàn (chỉ
    trả về khi phần ĐẦU thật sự là JSON hợp lệ, không đoán mò nội dung)."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except json.JSONDecodeError:
        return None


def _repair_invalid_single_quote_escape(text: str) -> str:
    """`\\'` (backslash + nháy đơn) KHÔNG PHẢI escape hợp lệ trong JSON (chỉ \\" mới hợp lệ) --
    model hay tự ý thêm nó khi trích dẫn 1 từ (thói quen từ Python/JS) dù đã được dặn không cần.
    Chỉ thay ĐÚNG 2 ký tự "\\'" thành "'" (không đụng vào bất kỳ escape hợp lệ nào khác, kể cả
    \\\\' -- backslash thật rồi mới đến nháy đơn -- vì chuỗi cần thay là "\\" + "'" liền nhau,
    không match được bên trong "\\\\" + "'")."""
    return text.replace("\\'", "'")


def _strip_open_fence(text: str) -> str:
    """Bóc fence MỞ "```json"/"```" ở đầu (nếu có) và fence ĐÓNG ở cuối (nếu có) -- KHÔNG đòi
    phải có ĐỦ CẢ HAI như _JSON_FENCE_RE. Dùng làm fallback cho trường hợp model MỞ fence nhưng
    quên đóng (response vẫn đủ dài/đầy đủ nội dung JSON, chỉ thiếu 3 ký tự "```" cuối cùng --
    raw_decode() bên dưới vẫn parse đúng vì JSON tự nó có cặp {}/[] cân bằng, không cần fence)."""
    text = text.strip()
    text = _LEADING_FENCE_RE.sub("", text, count=1)
    text = _TRAILING_FENCE_RE.sub("", text, count=1)
    return text.strip()


def extract_knowledge_json(response_text: str) -> Optional[dict]:
    """Lấy khối ```json ... ``` CUỐI CÙNG trong response (nếu model lỡ in ra nhiều khối, khối
    cuối luôn là bản họ muốn giữ lại). Nếu regex đòi CẢ 2 fence không match được (model quên fence
    đóng, hoặc không có fence nào cả), fallback sang bóc fence MỞ một mình (hoặc không bóc gì nếu
    không có), rồi mới đến response gốc y nguyên -- mỗi candidate đều thử raw_decode() (khoan
    dung text thừa phía sau) VÀ thử lại sau khi sửa escape "\\'" sai nếu lần đầu vẫn fail."""
    matches = _JSON_FENCE_RE.findall(response_text)
    candidates = [matches[-1].strip()] if matches else []
    candidates.append(_strip_open_fence(response_text))
    candidates.append(response_text.strip())

    for candidate in candidates:
        parsed = _loads_lenient(candidate)
        if parsed is None:
            parsed = _loads_lenient(_repair_invalid_single_quote_escape(candidate))
        if parsed is not None:
            return parsed
    return None


def extract_consensus_vote(response_text: str) -> Optional[str]:
    m = _CONSENSUS_MARKER_RE.search(response_text)
    return m.group(1).upper() if m else None


def describe_json_error(response_text: str) -> str:
    """Tái hiện ĐÚNG logic extract_knowledge_json() (kể cả các fallback bóc fence mở/sửa escape
    "\\'") để báo LỖI CUỐI CÙNG THẬT SỰ khiến TẤT CẢ candidate đều fail -- dùng candidate cuối
    cùng (đầy đủ nội dung nhất, sau khi đã bóc mọi fence có thể) để dòng/cột báo ra khớp với
    những gì người đọc thấy trong response gốc."""
    matches = _JSON_FENCE_RE.findall(response_text)
    candidates = [matches[-1].strip()] if matches else []
    candidates.append(_strip_open_fence(response_text))
    candidates.append(response_text.strip())

    # Ghi nhận lỗi có e.pos LỚN NHẤT (candidate/lượt sửa nào parse được XA NHẤT trước khi gãy) --
    # KHÔNG phải lỗi cuối cùng thử được, vì các candidate sau (ví dụ response gốc còn nguyên cả
    # dấu fence "```") luôn fail ngay tại ký tự 0, che mất lỗi thật sự nằm sâu hơn ở candidate
    # đã bóc fence.
    best_error = None
    best_candidate = candidates[-1] if candidates else response_text
    for candidate in candidates:
        for text in (candidate, _repair_invalid_single_quote_escape(candidate)):
            try:
                json.loads(text)
                return "Parse lại thành công (?) -- không tái hiện được lỗi."
            except json.JSONDecodeError as e:
                if best_error is None or e.pos > best_error.pos:
                    best_error, best_candidate = e, text

    start = max(0, best_error.pos - 150)
    end = min(len(best_candidate), best_error.pos + 150)
    return (
        f"JSONDecodeError: {best_error.msg} tại dòng {best_error.lineno}, cột {best_error.colno} "
        f"(ký tự thứ {best_error.pos}/{len(best_candidate)}).\n"
        f"    --- 150 ký tự TRƯỚC/SAU vị trí lỗi ---\n{best_candidate[start:end]!r}"
    )


def build_turn_prompt(task: str, variant: Optional[str], seed: dict, state: RelayState, model: str) -> str:
    """Chưa có knowledge nào parse thành công (state.latest_knowledge is None -- KHÔNG phải chỉ
    kiểm tra "turn đầu tiên", vì 1 turn có thể lỗi/không parse được, và nếu lượt SAU đó vẫn dựa
    theo "đã có turn trước" thì sẽ mất hẳn evidence gốc, model phải bịa từ đầu): instructions +
    schema + evidence lấy nguyên từ seed -- evidence là "candidate_units" (nhóm theo sample, giữ
    ngữ cảnh multi-term) nếu seed có, else "candidate_words" (phẳng theo từ, dùng cho 4 task cũ).
    Đã có knowledge (bất kể turn trước thành công hay không): tóm tắt NGẮN turn ngay trước + TOÀN
    BỘ knowledge JSON hiện tại (không phải diff)."""
    header = f"[Relay tri thức tự động -- task={task}" + (f", variant={variant}" if variant else "") + f"] Lượt cho model: {model.upper()}\n"

    prev = state.turns[-1] if state.turns else None
    prev_note = (
        f"Model trước ({prev.model}) vừa sửa knowledge graph và vote "
        f"CONSENSUS: {prev.consensus_vote or 'CONTINUE (không rõ, thiếu marker/lỗi gọi API)'}.\n\n"
        if prev is not None else ""
    )

    if state.latest_knowledge is None:
        if "candidate_units" in seed:
            evidence_block = (
                f"DANH SÁCH SAMPLE (mỗi sample có transcript + TOÀN BỘ thuật ngữ xuất hiện CÙNG "
                f"câu đó, để giữ ngữ cảnh liên hệ giữa các thuật ngữ):\n"
                f"{json.dumps(seed['candidate_units'], ensure_ascii=False, indent=2)}"
            )
        else:
            evidence_block = (
                f"DANH SÁCH TỪ CANDIDATE (đã gộp từ CSV hiện có + web-research + quét corpus, "
                f"kèm bằng chứng thật nếu có):\n{json.dumps(seed['candidate_words'], ensure_ascii=False, indent=2)}\n\n"
                f"MỘT SỐ TRANSCRIPT CHƯA KHỚP TỪ NÀO (tham khảo để đề xuất từ MỚI nếu bạn thấy phù hợp):\n"
                f"{json.dumps(seed.get('unmatched_transcript_sample', []), ensure_ascii=False, indent=2)}"
            )
        body = (
            f"{prev_note}"
            f"{seed['instructions']}\n\n"
            f"SCHEMA đích (trả về ĐÚNG dạng này):\n{seed['schema_spec']}\n\n"
            f"{evidence_block}"
        )
    else:
        body = (
            f"{prev_note}"
            f"{seed['instructions']}\n\n"
            f"SCHEMA đích:\n{seed['schema_spec']}\n\n"
            f"KNOWLEDGE GRAPH HIỆN TẠI (hãy rà soát/sửa/thêm/bớt, rồi trả lại TOÀN BỘ, không phải diff):\n"
            f"{json.dumps(state.latest_knowledge, ensure_ascii=False, indent=2)}"
        )

    return header + "\n" + body


def load_gemini_client(service_account_path: str, *, location: str = "global"):
    """Client Vertex AI THẬT (google.genai SDK) -- dùng khi có --gemini-service-account-json
    (điển hình: đang chạy trong Colab). Phân biệt với client OpenAI-compatible (dùng khi chạy
    local qua .env, kể cả CHO Gemini nếu trỏ vào 1 proxy local nói giao thức OpenAI) bằng
    _is_openai_style_client() trong call_model() bên dưới -- không cần biết trước đang ở env nào."""
    from google import genai
    from google.oauth2 import service_account

    with open(service_account_path, encoding="utf-8") as f:
        info = json.load(f)
    project_id = info["project_id"]

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)


DEFAULT_REQUEST_TIMEOUT_SECONDS = 300


def _make_openai_client(api_key: str, base_url: str, *, timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS):
    from openai import OpenAI

    # KHÔNG đặt timeout -> 1 request treo (proxy local/tunnel không phản hồi) sẽ chờ VÔ HẠN,
    # không bao giờ vào nhánh except để retry/bỏ qua lượt -- cả relay đứng im, không log gì.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def load_openai_client(api_key_path: Path, base_url: str):
    raw = Path(api_key_path).read_text(encoding="utf-8").strip()
    # Chấp nhận cả 2 dạng: key trần trên 1 dòng, HOẶC dòng kiểu env-var
    # OPENAI_API_KEY="..."/OPENAI_API_KEY=... -- lấy đúng phần giá trị, bỏ dấu nháy nếu có.
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    api_key = raw.strip('"').strip("'")
    return _make_openai_client(api_key, base_url)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def load_env_file(path: Path) -> dict[str, dict[str, Optional[str]]]:
    """Parse 1 file .env dạng đặc thù đang dùng cho chạy local:
        OPENAI_API_KEY="..." # https://r3wrrfi.abc-tunnel.us/v1 # cx/gpt-5.6-luna
        GEMINI_API_KEY="..." # http://localhost:20128/v1 # ag/gemini-3.5-flash-extra-low
    Mỗi dòng KEY="value" [# base_url [# model]] -- base_url/model nằm trong COMMENT (không phải
    biến env chuẩn) vì đây là quy ước riêng của người dùng, không phải cú pháp .env thông thường.
    Trả về {"OPENAI": {"api_key":..., "base_url":..., "model":...}, "GEMINI": {...}} (bỏ qua dòng
    trống/bắt đầu bằng "#" thuần, không hỗ trợ các biến env khác ngoài *_API_KEY)."""
    result: dict[str, dict[str, Optional[str]]] = {}
    if not path.exists():
        return result

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key_part, _, rest = line.partition("=")
        key_part = key_part.strip()
        if not key_part.endswith("_API_KEY"):
            continue
        prefix = key_part[: -len("_API_KEY")]  # "OPENAI" | "GEMINI" | ...

        value_part, _, comment_part = rest.partition("#")
        api_key = value_part.strip().strip('"').strip("'")
        comment_fields = [c.strip() for c in comment_part.split("#")] if comment_part else []
        base_url = comment_fields[0] if len(comment_fields) >= 1 and comment_fields[0] else None
        model = comment_fields[1] if len(comment_fields) >= 2 and comment_fields[1] else None

        result[prefix] = {"api_key": api_key or None, "base_url": base_url, "model": model}
    return result


DEFAULT_MAX_OUTPUT_TOKENS = 32768


def _is_openai_style_client(client) -> bool:
    return hasattr(client, "chat") and hasattr(client.chat, "completions")


def call_model(
    model_name: str, prompt: str, *,
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    system_instruction: Optional[str] = None, temperature: float = 0.7,
) -> str:
    """model_name == "gemini" -> dùng gemini_client/gemini_model; NGƯỢC LẠI (mọi tên khác, ví dụ
    "openai") -> dùng openai_client/openai_model. Client THẬT SỰ gọi ra sao (Vertex AI SDK hay
    OpenAI-compatible HTTP) được TỰ NHẬN DIỆN qua _is_openai_style_client() -- KHÔNG cần biết
    trước đang chạy Colab (Vertex service account) hay local (.env, kể cả Gemini qua 1 proxy local
    nói giao thức OpenAI) -- chỉ cần đưa ĐÚNG loại client object vào, code tự xử lý đúng cách.
    Hàm dùng chung cho CẢ debate relay (không system_instruction, prompt tự chứa mọi thứ) LẪN các
    pipeline khác (Code-switching classify-cs/generate-questions/filter-questions) muốn tách
    system prompt riêng, giống hệt cách các pipeline Hán Việt/Từ mượn/... vẫn dùng system prompt.

    max_output_tokens mặc định CAO (mỗi lượt phải trả về TOÀN BỘ knowledge graph của cả batch,
    không phải diff -- batch càng lớn/càng nhiều field thì response càng dài; response bị cắt
    cụt giữa chừng sẽ KHÔNG parse được JSON, gây lỗi "Không parse được khối JSON hợp lệ")."""
    client = gemini_client if model_name == "gemini" else openai_client
    model = gemini_model if model_name == "gemini" else openai_model

    if _is_openai_style_client(client):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""

    if model_name == "gemini":
        from google.genai import types

        response = gemini_client.models.generate_content(
            model=gemini_model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            ),
        )
        return response.text or ""

    raise TypeError(
        f"Client cho model_name={model_name!r} không phải Vertex AI genai.Client cũng không "
        "phải OpenAI-compatible client -- không biết cách gọi."
    )


def _turn_to_dict(t: Turn) -> dict:
    return asdict(t)


def _write_turn_log(log_dir: Path, turn: Turn) -> None:
    path = log_dir / f"turn_{turn.turn_index:03d}_{turn.model}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_turn_to_dict(turn), f, ensure_ascii=False, indent=2)


_SKIP_SENTINEL = "SKIP"


def manual_turn_io(
    prompt: str, *, prompt_path: Path, response_path: Path,
    input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
) -> str:
    """Khôi phục cơ chế copy-paste thủ công (manual_model_relay.py cũ, đã bị thay bằng API):
    ghi prompt vào prompt_path (KHÔNG in ra terminal -- có thể dài hàng chục KB), xoá
    response_path CŨ (tránh đọc nhầm câu trả lời của lượt trước), hỏi xác nhận NGẮN qua
    input_fn() (Enter khi đã dán xong, gõ SKIP để bỏ lượt nếu paste hỏng), rồi đọc response_path.
    Trả về "" nếu SKIP -- run_turn() coi đây như 1 lượt lỗi thông thường (parsed_knowledge=None,
    relay vẫn tiếp tục), không phải lỗi hệ thống."""
    prompt_path.write_text(prompt, encoding="utf-8")
    if response_path.exists():
        response_path.unlink()

    print_fn(f"\n>>> Mở {prompt_path}, copy TOÀN BỘ nội dung, dán vào model (web).")
    print_fn(f">>> Dán câu trả lời của model vào {response_path} rồi lưu file lại.")

    while True:
        answer = input_fn(f"Đã dán xong câu trả lời vào {response_path.name}? (Enter để đọc tiếp, gõ {_SKIP_SENTINEL} để bỏ qua lượt này): ")
        if answer.strip().upper() == _SKIP_SENTINEL:
            return ""
        if not response_path.exists() or not response_path.read_text(encoding="utf-8").strip():
            print_fn(f"[LỖI] {response_path} vẫn trống -- dán câu trả lời vào file rồi Enter lại, hoặc gõ {_SKIP_SENTINEL}.")
            continue
        return response_path.read_text(encoding="utf-8")


def run_turn(
    task: str, variant: Optional[str], seed: dict, state: RelayState, model: str, *,
    debate_mode: str = "api",
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_path: Optional[Path] = None, response_path: Optional[Path] = None,
    input_fn: Callable[[str], str] = input,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print,
) -> Turn:
    """debate_mode="api" (mặc định): soạn prompt rồi gọi API NGAY để lấy response -- không cần
    người dán tay. debate_mode="manual": soạn prompt rồi ghi ra prompt_path/response_path, chờ
    người copy-paste thủ công qua manual_turn_io() (KHÔNG retry API, KHÔNG cần max_retries> 1 --
    người dùng tự sửa/dán lại nếu JSON hỏng, vòng while trong manual_turn_io() đã xử lý việc đó).
    Retry (kèm backoff, CHỈ áp dụng debate_mode="api") khi lỗi gọi API hoặc response không parse
    được JSON hợp lệ; sau max_retries lần thất bại, ghi nhận 1 turn "trắng"
    (parsed_knowledge=None) và relay tiếp tục (không dừng cả relay vì 1 lượt lỗi)."""
    prompt = build_turn_prompt(task, variant, seed, state, model)
    round_index = len(state.turns) // len(state.model_order) + 1
    turn_index = len(state.turns) + 1

    if debate_mode == "manual":
        response = manual_turn_io(
            prompt, prompt_path=prompt_path, response_path=response_path,
            input_fn=input_fn, print_fn=print_fn,
        )
        parsed = extract_knowledge_json(response) if response else None
        vote = extract_consensus_vote(response) if response else None
        turn = Turn(round_index, turn_index, model, prompt, response, parsed, vote)
        state.turns.append(turn)
        if parsed is not None:
            state.latest_knowledge = parsed
        if log_dir is not None:
            _write_turn_log(log_dir, turn)
        return turn

    last_error = None
    last_response = None
    for attempt in range(max_retries):
        try:
            response = call_model(
                model, prompt, gemini_client=gemini_client, gemini_model=gemini_model,
                openai_client=openai_client, openai_model=openai_model,
                max_output_tokens=max_output_tokens,
            )
            last_response = response
            parsed = extract_knowledge_json(response)
            if parsed is None:
                print_fn(
                    f"[DEBUG] lượt {model} (round {round_index}, thử {attempt + 1}/{max_retries}) "
                    f"không parse được JSON -- response dài {len(response)} ký tự.\n"
                    f"    {describe_json_error(response)}"
                )
                raise ValueError("Không parse được khối JSON hợp lệ trong response")
            vote = extract_consensus_vote(response)
            turn = Turn(round_index, turn_index, model, prompt, response, parsed, vote)
            state.turns.append(turn)
            state.latest_knowledge = parsed
            if log_dir is not None:
                _write_turn_log(log_dir, turn)
            return turn
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print_fn(f"[LỖI] lượt {model} (round {round_index}) thất bại sau {max_retries} lần thử: {last_error!r} -- bỏ qua lượt này.")
    turn = Turn(round_index, turn_index, model, prompt, last_response or "", None, None)
    state.turns.append(turn)
    if log_dir is not None:
        _write_turn_log(log_dir, turn)
    return turn


def run_round(
    task: str, variant: Optional[str], seed: dict, state: RelayState, *,
    debate_mode: str = "api",
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_path: Optional[Path] = None, response_path: Optional[Path] = None,
    input_fn: Callable[[str], str] = input,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print,
) -> bool:
    """Chạy đủ len(state.model_order) turn. Trả True nếu CẢ round này mọi turn đều vote FINAL."""
    votes = []
    for model in state.model_order:
        turn = run_turn(
            task, variant, seed, state, model,
            debate_mode=debate_mode,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn,
            log_dir=log_dir, print_fn=print_fn,
        )
        print_fn(f"  round {turn.round_index} [{turn.model}] vote={turn.consensus_vote} "
                  f"parsed={'ok' if turn.parsed_knowledge else 'lỗi'}")
        votes.append(turn.consensus_vote)
    return bool(votes) and all(v == "FINAL" for v in votes)


def run_relay(
    task: str, seed: dict, *, variant: Optional[str] = None,
    model_order: Optional[list[str]] = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
    debate_mode: str = "api",
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_path: Optional[Path] = None, response_path: Optional[Path] = None,
    input_fn: Callable[[str], str] = input,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print, resume_state: Optional[RelayState] = None,
) -> RelayState:
    state = resume_state or RelayState(task=task, variant=variant, model_order=model_order or DEFAULT_MODEL_ORDER, max_rounds=max_rounds)

    rounds_done = len(state.turns) // len(state.model_order)
    for _ in range(rounds_done, state.max_rounds):
        consensus = run_round(
            task, variant, seed, state,
            debate_mode=debate_mode,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn,
            log_dir=log_dir, print_fn=print_fn,
        )
        if consensus:
            state.stopped_reason = "consensus"
            print_fn(f">>> Cả {len(state.model_order)} model đồng thuận CONSENSUS: FINAL -- dừng relay.")
            return state

    state.stopped_reason = "max_rounds"
    print_fn(f">>> Đã hết {state.max_rounds} vòng mà chưa đồng thuận -- dùng bản tri thức mới nhất hiện có.")
    return state


def save_state(state: RelayState, path: Path) -> None:
    payload = {
        "task": state.task, "variant": state.variant, "model_order": state.model_order,
        "max_rounds": state.max_rounds, "stopped_reason": state.stopped_reason,
        "latest_knowledge": state.latest_knowledge, "turns": [_turn_to_dict(t) for t in state.turns],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_state(path: Path) -> RelayState:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    turns = [Turn(**t) for t in payload["turns"]]
    return RelayState(
        task=payload["task"], variant=payload["variant"], model_order=payload["model_order"],
        max_rounds=payload["max_rounds"], turns=turns, latest_knowledge=payload["latest_knowledge"],
        stopped_reason=payload["stopped_reason"],
    )


def export_knowledge_graph(state: RelayState, task: str, variant: Optional[str], output_path: Path) -> None:
    """Ghi latest_knowledge + relay_meta của 1 RelayState (batch DUY NHẤT) -- dùng khi
    run_relay_batched() chỉ tạo ra 1 batch (batch_size=None, đúng hành vi cũ của 4 task hiện có).
    Format Y HỆT manual_model_relay.export_knowledge_graph cũ."""
    if state.latest_knowledge is None:
        raise ValueError("Chưa có knowledge graph nào được parse thành công -- không có gì để export.")

    consensus_round = None
    if state.stopped_reason == "consensus":
        consensus_round = state.turns[-1].round_index if state.turns else None

    kg = dict(state.latest_knowledge)
    kg["task"] = task
    kg["variant"] = variant
    kg["generated_at"] = datetime.now(timezone.utc).isoformat()
    kg["relay_meta"] = {
        "model_order": state.model_order,
        "rounds_completed": (len(state.turns) + len(state.model_order) - 1) // len(state.model_order),
        "max_rounds": state.max_rounds, "stopped_reason": state.stopped_reason,
        "consensus_round": consensus_round,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu knowledge graph vào {output_path}.")


# ============================================================================================
# Batching -- chỉ thật sự "chia batch" khi --batch-size được truyền; mặc định None -> 1 batch
# duy nhất chứa hết candidate, tức HÀNH VI Y HỆT single-shot cũ.
# ============================================================================================

def chunk_seed_units(seed: dict, batch_size: Optional[int]) -> list[dict]:
    """Chia seed thành nhiều seed con theo batch_size, áp dụng cho "candidate_units" (nếu có)
    else "candidate_words". batch_size=None hoặc <= tổng số item -> trả về [seed] nguyên vẹn."""
    key = "candidate_units" if "candidate_units" in seed else "candidate_words"
    items = seed.get(key, [])
    if not batch_size or len(items) <= batch_size:
        return [seed]
    return [
        {**seed, key: items[i:i + batch_size]}
        for i in range(0, len(items), batch_size)
    ]


def _batch_prompt_response_paths(
    prompt_path: Optional[Path], response_path: Optional[Path], i: int, n_batches: int,
) -> tuple[Optional[Path], Optional[Path]]:
    if n_batches == 1 or prompt_path is None or response_path is None:
        return prompt_path, response_path
    suffix = f"_batch{i:04d}"
    return (
        prompt_path.with_name(f"{prompt_path.stem}{suffix}{prompt_path.suffix}"),
        response_path.with_name(f"{response_path.stem}{suffix}{response_path.suffix}"),
    )


def run_relay_batched(
    task: str, seed: dict, *, variant: Optional[str] = None,
    model_order: Optional[list[str]] = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
    batch_size: Optional[int] = None, max_workers: int = 4,
    debate_mode: str = "api",
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_path: Optional[Path] = None, response_path: Optional[Path] = None,
    input_fn: Callable[[str], str] = input,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print, resume_states: Optional[list] = None,
) -> list[RelayState]:
    """batch_size=None -> chạy y hệt run_relay() với seed gốc (1 phần tử list trả về) -- giữ
    nguyên hành vi 4 task cũ. batch_size được set -> chia batch. debate_mode="api": các batch
    chạy SONG SONG (mỗi batch tự tuần tự Gemini<->OpenAI bên trong). debate_mode="manual": LUÔN
    chạy TUẦN TỰ từng batch (người dùng không thể copy-paste nhiều batch cùng lúc), mỗi batch
    dùng 1 cặp prompt/response file RIÊNG (hậu tố "_batch<i>")."""
    batches = chunk_seed_units(seed, batch_size)
    resume_states = resume_states or [None] * len(batches)

    if len(batches) == 1:
        state = run_relay(
            task, batches[0], variant=variant, model_order=model_order, max_rounds=max_rounds,
            debate_mode=debate_mode,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn,
            log_dir=log_dir, print_fn=print_fn, resume_state=resume_states[0],
        )
        return [state]

    if debate_mode == "manual":
        states: list = []
        for i, batch_seed in enumerate(batches):
            print_fn(f"\n===== Batch {i + 1}/{len(batches)} =====")
            batch_log_dir = (log_dir / f"batch_{i:04d}") if log_dir else None
            if batch_log_dir:
                batch_log_dir.mkdir(parents=True, exist_ok=True)
            batch_prompt_path, batch_response_path = _batch_prompt_response_paths(prompt_path, response_path, i, len(batches))
            states.append(run_relay(
                task, batch_seed, variant=variant, model_order=model_order, max_rounds=max_rounds,
                debate_mode="manual",
                prompt_path=batch_prompt_path, response_path=batch_response_path, input_fn=input_fn,
                max_retries=max_retries, log_dir=batch_log_dir, print_fn=print_fn, resume_state=resume_states[i],
            ))
        return states

    # Nhiều batch chạy song song (debate_mode="api") -> log "round X [model] vote=..." của TỪNG
    # batch xen kẽ nhau, gần như không đọc được (đây chính là hàng trăm dòng người dùng thấy khi
    # debug tốc độ) -- khi batch hoá, LỌC BỚT chỉ còn dòng lỗi/debug/kết luận (">>> ", "[LỖI]",
    # "[DEBUG]") + 1 thanh progress bar theo SỐ BATCH đã xong (dễ theo dõi tiến độ thật hơn nhiều
    # so với log thô). Người gọi vẫn có thể truyền print_fn khác để giữ nguyên log chi tiết.
    def _filtered_print_fn(msg: str) -> None:
        if msg.startswith(("[LỖI]", "[DEBUG]", ">>>")):
            print_fn(msg)

    def _run_one(i: int, batch_seed: dict) -> RelayState:
        batch_log_dir = (log_dir / f"batch_{i:04d}") if log_dir else None
        if batch_log_dir:
            batch_log_dir.mkdir(parents=True, exist_ok=True)
        return run_relay(
            task, batch_seed, variant=variant, model_order=model_order, max_rounds=max_rounds,
            debate_mode="api",
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            log_dir=batch_log_dir, print_fn=_filtered_print_fn, resume_state=resume_states[i],
        )

    states: list = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, i, b): i for i, b in enumerate(batches)}
        with tqdm(total=len(batches), desc=f"debate {task} ({len(batches)} batch, {max_workers} luồng)") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                states[i] = future.result()
                pbar.update(1)
    return states


def _field_richness(entry: dict) -> int:
    return sum(1 for v in (entry.get("fields") or {}).values() if v not in (None, "", []))


def merge_batch_knowledge(states: list[RelayState]) -> tuple[dict, int]:
    """Gộp "words" (dedupe theo "word", ưu tiên bản có NHIỀU field khác rỗng hơn khi trùng),
    "sample_relations" (nối thẳng -- khoá theo sample_id vốn không trùng giữa các batch), "rules"
    (mỗi key giữ description DÀI NHẤT tìm được). Trả về (merged_dict, số batch đã đồng thuận)."""
    merged_words: dict[str, dict] = {}
    merged_relations: list[dict] = []
    merged_rules: dict[str, dict] = {}
    n_consensus = 0

    for state in states:
        if state is None or state.latest_knowledge is None:
            continue
        if state.stopped_reason == "consensus":
            n_consensus += 1
        kg = state.latest_knowledge
        for entry in kg.get("words", []) or []:
            word = entry["word"]
            if word not in merged_words or _field_richness(entry) > _field_richness(merged_words[word]):
                merged_words[word] = entry
        merged_relations.extend(kg.get("sample_relations", []) or [])
        for key, rule in (kg.get("rules") or {}).items():
            desc = rule.get("description") or ""
            if key not in merged_rules or len(desc) > len(merged_rules[key].get("description") or ""):
                merged_rules[key] = rule

    return {
        "words": list(merged_words.values()),
        "sample_relations": merged_relations,
        "rules": merged_rules,
    }, n_consensus


def export_merged_knowledge_graph(
    states: list[RelayState], task: str, variant: Optional[str], output_path: Path, *,
    model_order: list[str], max_rounds: int,
) -> None:
    merged, n_consensus = merge_batch_knowledge(states)
    if not merged["words"] and not merged["sample_relations"]:
        raise ValueError("Không batch nào có knowledge hợp lệ để export.")

    kg = dict(merged)
    kg["schema_version"] = SCHEMA_VERSION
    kg["task"] = task
    kg["variant"] = variant
    kg["generated_at"] = datetime.now(timezone.utc).isoformat()
    kg["relay_meta"] = {
        "model_order": model_order, "max_rounds": max_rounds,
        "n_batches": len(states), "n_consensus": n_consensus,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu knowledge graph gộp từ {len(states)} batch ({n_consensus} batch đồng thuận) vào {output_path}.")


def _batch_state_path(state_output: Path, i: int) -> Path:
    return state_output.with_name(f"{state_output.stem}_batch{i:04d}{state_output.suffix}")


def save_batch_states(states: list[RelayState], state_output: Path) -> None:
    """1 batch -> ghi state_output y hệt save_state() cũ. Nhiều batch -> ghi 1 file trạng thái
    riêng cho mỗi batch (để có thể --resume từng batch độc lập) CỘNG 1 file tóm tắt tại chính
    state_output (đọc được bởi subcommand "show")."""
    if len(states) == 1:
        save_state(states[0], state_output)
        return

    for i, state in enumerate(states):
        save_state(state, _batch_state_path(state_output, i))

    summary = {
        "n_batches": len(states),
        "task": states[0].task, "variant": states[0].variant,
        "model_order": states[0].model_order, "max_rounds": states[0].max_rounds,
        "batch_summaries": [
            {"batch_index": i, "n_turns": len(s.turns), "stopped_reason": s.stopped_reason}
            for i, s in enumerate(states)
        ],
    }
    with open(state_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_batch_states(state_output: Path, n_batches: int) -> list[Optional[RelayState]]:
    result: list[Optional[RelayState]] = []
    for i in range(n_batches):
        p = _batch_state_path(state_output, i)
        result.append(load_state(p) if p.exists() else None)
    return result


def resolve_clients(args) -> tuple:
    """TỰ NHẬN DIỆN nguồn credential cho cả Gemini lẫn OpenAI, không cần biết trước đang chạy
    Colab hay local: ưu tiên --gemini-service-account-json (Vertex AI thật, --openai-api-key-file
    (file key riêng) nếu được truyền; nếu KHÔNG, rơi về --env-file (mặc định .env cạnh
    --seed-json nếu không truyền --env-file) -- CẢ Gemini lẫn OpenAI khi đọc từ .env đều dùng
    chung 1 client OpenAI-compatible (call_model() tự nhận diện qua _is_openai_style_client(),
    không cần phân biệt ở đây). --gemini-model/--openai-model/--*-base-url truyền tay LUÔN được
    ưu tiên cao nhất nếu có, kể cả khi cũng lấy key từ .env."""
    if args.env_file:
        env_path = Path(args.env_file)
    elif args.location == "local":
        env_path = env_paths.default_env_file(args.location)
    else:
        env_path = Path(args.seed_json).resolve().parent / ".env"
    env = load_env_file(env_path) if env_path.exists() else {}

    gemini_service_account_json = args.gemini_service_account_json
    if not gemini_service_account_json and not args.env_file and args.location == "drive" and "GEMINI" not in env:
        gemini_service_account_json = str(env_paths.gemini_service_account_path(args.location))

    if gemini_service_account_json:
        gemini_client = load_gemini_client(gemini_service_account_json)
        gemini_model = args.gemini_model or DEFAULT_GEMINI_MODEL
    else:
        gemini_env = env.get("GEMINI", {})
        gemini_api_key = gemini_env.get("api_key")
        gemini_base_url = args.gemini_base_url or gemini_env.get("base_url")
        gemini_model = args.gemini_model or gemini_env.get("model") or DEFAULT_GEMINI_MODEL
        if not (gemini_api_key and gemini_base_url):
            raise ValueError(
                f"Không có --gemini-service-account-json, và không tìm được GEMINI_API_KEY + "
                f"base_url hợp lệ trong {env_path} -- truyền 1 trong 2 cách cấu hình Gemini."
            )
        gemini_client = _make_openai_client(gemini_api_key, gemini_base_url)

    if args.openai_api_key_file:
        openai_client = load_openai_client(Path(args.openai_api_key_file), args.openai_base_url or DEFAULT_OPENAI_BASE_URL)
        openai_model = args.openai_model or DEFAULT_OPENAI_MODEL
    else:
        openai_env = env.get("OPENAI", {})
        openai_api_key = openai_env.get("api_key")
        openai_base_url = args.openai_base_url or openai_env.get("base_url")
        openai_model = args.openai_model or openai_env.get("model") or DEFAULT_OPENAI_MODEL
        if not (openai_api_key and openai_base_url):
            raise ValueError(
                f"Không có --openai-api-key-file, và không tìm được OPENAI_API_KEY + base_url "
                f"hợp lệ trong {env_path} -- truyền 1 trong 2 cách cấu hình OpenAI."
            )
        openai_client = _make_openai_client(openai_api_key, openai_base_url)

    return gemini_client, gemini_model, openai_client, openai_model


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("run", help="Chạy relay tri thức (debate-mode api: tự động qua API; manual: copy-paste thủ công).")
    p1.add_argument("--task", required=True, help="Tên task tự do (không giới hạn danh sách cố định) -- dùng để tra INSTRUCTIONS_BY_TASK trong build_debate_seed.py.")
    p1.add_argument("--variant", default=None)
    p1.add_argument("--debate-mode", choices=["api", "manual"], default="api", help='"api" (mặc định, tự động qua Gemini+OpenAI-compatible) hoặc "manual" (copy-paste thủ công qua file, không cần credential nào).')
    env_paths.add_location_arg(p1)
    p1.add_argument("--seed-json", required=True)
    p1.add_argument("--state-output", default=None, help="Mặc định: {knowledge_dir(--location)}/relay_state_<task>[_<variant>].json.")
    p1.add_argument("--knowledge-output", default=None, help="Mặc định: {knowledge_dir(--location)}/knowledge_<task>[_<variant>].json.")
    p1.add_argument("--prompt-file", default=None, help='Chỉ dùng khi --debate-mode manual. Mặc định: "relay_prompt_<task>[_<variant>].txt" cạnh --state-output.')
    p1.add_argument("--response-file", default=None, help='Chỉ dùng khi --debate-mode manual. Mặc định: "relay_response_<task>[_<variant>].txt" cạnh --state-output.')
    p1.add_argument("--env-file", default=None, help='Chỉ dùng khi --debate-mode api. File .env local dạng KEY="value" # base_url # model (xem load_env_file()) -- dùng khi KHÔNG truyền --gemini-service-account-json/--openai-api-key-file, cho cả Gemini (kể cả qua proxy local nói giao thức OpenAI) lẫn OpenAI. Mặc định: env_paths.default_env_file(--location) khi --location local.')
    p1.add_argument("--gemini-service-account-json", default=None, help='Chỉ dùng khi --debate-mode api. Vertex AI service account JSON. Mặc định: env_paths.gemini_service_account_path(--location) khi --location drive.')
    p1.add_argument("--gemini-model", default=None)
    p1.add_argument("--gemini-base-url", default=None, help="Chỉ dùng khi KHÔNG có --gemini-service-account-json và muốn override base_url thay vì lấy từ --env-file.")
    p1.add_argument("--openai-api-key-file", default=None, help="File chứa OpenAI-compatible API key. Bỏ trống để lấy OPENAI_API_KEY/base_url/model từ --env-file.")
    p1.add_argument("--openai-base-url", default=None)
    p1.add_argument("--openai-model", default=None)
    p1.add_argument("--model-order", default=None, help='Danh sách model, phân tách bởi dấu phẩy. Mặc định: "gemini,openai" (api) hoặc "gemini,chatgpt,claude" (manual).')
    p1.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p1.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Tăng nếu batch lớn/knowledge dài bị cắt cụt (lỗi 'Không parse được khối JSON hợp lệ'). Không áp dụng --debate-mode manual.")
    p1.add_argument("--batch-size", type=int, default=None, help="Chia candidate thành nhiều batch (mặc định None = 1 batch duy nhất, đúng hành vi cũ).")
    p1.add_argument("--max-workers", type=int, default=4, help="Số batch chạy song song (chỉ áp dụng --debate-mode api -- manual luôn chạy tuần tự).")
    p1.add_argument("--log-dir", default=None, help="Optional: ghi lại prompt/response từng lượt để audit.")
    p1.add_argument("--resume", action="store_true", help="Tiếp tục từ --state-output (hoặc các file batch tương ứng) nếu đã tồn tại.")

    p2 = sub.add_parser("show", help="In lại trạng thái relay đã lưu (không chạy lại).")
    p2.add_argument("--state", required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        with open(args.seed_json, encoding="utf-8") as f:
            seed = json.load(f)

        suffix = f"_{args.task}" + (f"_{args.variant}" if args.variant else "")
        state_path = Path(args.state_output) if args.state_output else env_paths.knowledge_dir(args.location) / f"relay_state{suffix}.json"
        knowledge_output = Path(args.knowledge_output) if args.knowledge_output else env_paths.knowledge_dir(args.location) / f"knowledge{suffix}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        gemini_client = gemini_model = openai_client = openai_model = None
        prompt_path = response_path = None
        input_fn = input
        if args.debate_mode == "api":
            gemini_client, gemini_model, openai_client, openai_model = resolve_clients(args)
        else:
            prompt_path = Path(args.prompt_file) if args.prompt_file else state_path.with_name(f"relay_prompt{suffix}.txt")
            response_path = Path(args.response_file) if args.response_file else state_path.with_name(f"relay_response{suffix}.txt")

        log_dir = Path(args.log_dir) if args.log_dir else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        default_model_order = DEFAULT_MANUAL_MODEL_ORDER if args.debate_mode == "manual" else DEFAULT_MODEL_ORDER
        model_order = args.model_order.split(",") if args.model_order else default_model_order

        n_batches_guess = len(chunk_seed_units(seed, args.batch_size))
        resume_states = None
        if args.resume:
            if n_batches_guess == 1:
                resume_states = [load_state(state_path)] if state_path.exists() else [None]
            else:
                resume_states = load_batch_states(state_path, n_batches_guess)

        states = run_relay_batched(
            args.task, seed, variant=args.variant, model_order=model_order, max_rounds=args.max_rounds,
            batch_size=args.batch_size, max_workers=args.max_workers,
            debate_mode=args.debate_mode,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=args.max_retries, max_output_tokens=args.max_output_tokens,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn,
            log_dir=log_dir, resume_states=resume_states,
        )
        save_batch_states(states, state_path)

        if len(states) == 1:
            export_knowledge_graph(states[0], args.task, args.variant, knowledge_output)
        else:
            export_merged_knowledge_graph(
                states, args.task, args.variant, knowledge_output,
                model_order=model_order, max_rounds=args.max_rounds,
            )

    elif args.command == "show":
        with open(args.state, encoding="utf-8") as f:
            raw = json.load(f)

        if "n_batches" in raw:
            print(f"task={raw['task']} variant={raw['variant']} n_batches={raw['n_batches']}")
            for b in raw["batch_summaries"]:
                print(f"  batch {b['batch_index']}: {b['n_turns']} turns, stopped_reason={b['stopped_reason']}")
        else:
            state = load_state(Path(args.state))
            print(f"task={state.task} variant={state.variant} stopped_reason={state.stopped_reason} "
                  f"turns={len(state.turns)}/{state.max_rounds * len(state.model_order)}")
            for t in state.turns:
                print(f"  round {t.round_index} turn {t.turn_index} [{t.model}] vote={t.consensus_vote} "
                      f"parsed={'ok' if t.parsed_knowledge else 'lỗi'}")


if __name__ == "__main__":
    main()
