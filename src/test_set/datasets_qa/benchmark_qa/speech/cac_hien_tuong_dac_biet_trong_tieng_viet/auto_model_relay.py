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

from knowledge_graph import SCHEMA_VERSION

DEFAULT_MODEL_ORDER = ["gemini", "openai"]
DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OPENAI_MODEL = "cx/gpt-5.6-luna"
DEFAULT_OPENAI_BASE_URL = "https://r3wrrfi.abc-tunnel.us/v1"

_CONSENSUS_MARKER_RE = re.compile(r"CONSENSUS:\s*(FINAL|CONTINUE)", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


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


def extract_knowledge_json(response_text: str) -> Optional[dict]:
    """Lấy khối ```json ... ``` CUỐI CÙNG trong response (nếu model lỡ in ra nhiều khối, khối
    cuối luôn là bản họ muốn giữ lại). Nếu KHÔNG có khối fence nào, thử parse TOÀN BỘ
    response_text như 1 JSON object độc lập trước khi bỏ cuộc."""
    matches = _JSON_FENCE_RE.findall(response_text)
    if matches:
        try:
            return json.loads(matches[-1].strip())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None


def extract_consensus_vote(response_text: str) -> Optional[str]:
    m = _CONSENSUS_MARKER_RE.search(response_text)
    return m.group(1).upper() if m else None


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
    from google import genai
    from google.oauth2 import service_account

    with open(service_account_path, encoding="utf-8") as f:
        info = json.load(f)
    project_id = info["project_id"]

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)


def load_openai_client(api_key_path: Path, base_url: str):
    from openai import OpenAI

    raw = Path(api_key_path).read_text(encoding="utf-8").strip()
    # Chấp nhận cả 2 dạng: key trần trên 1 dòng, HOẶC dòng kiểu env-var
    # OPENAI_API_KEY="..."/OPENAI_API_KEY=... -- lấy đúng phần giá trị, bỏ dấu nháy nếu có.
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    api_key = raw.strip('"').strip("'")
    return OpenAI(api_key=api_key, base_url=base_url)


DEFAULT_MAX_OUTPUT_TOKENS = 32768


def call_model(
    model_name: str, prompt: str, *,
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """model_name == "gemini" -> gọi Gemini API; NGƯỢC LẠI (mọi tên khác, ví dụ "openai") -> gọi
    model OpenAI-compatible qua base_url riêng. Đây là điểm THAY THẾ DUY NHẤT so với
    manual_model_relay.py: lấy response NGAY LẬP TỨC qua API, không cần người dán tay.

    max_output_tokens mặc định CAO (mỗi lượt phải trả về TOÀN BỘ knowledge graph của cả batch,
    không phải diff -- batch càng lớn/càng nhiều field thì response càng dài; response bị cắt
    cụt giữa chừng sẽ KHÔNG parse được JSON, gây lỗi "Không parse được khối JSON hợp lệ")."""
    if model_name == "gemini":
        from google.genai import types

        response = gemini_client.models.generate_content(
            model=gemini_model, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=max_output_tokens),
        )
        return response.text or ""

    response = openai_client.chat.completions.create(
        model=openai_model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content or ""


def _turn_to_dict(t: Turn) -> dict:
    return asdict(t)


def _write_turn_log(log_dir: Path, turn: Turn) -> None:
    path = log_dir / f"turn_{turn.turn_index:03d}_{turn.model}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_turn_to_dict(turn), f, ensure_ascii=False, indent=2)


def run_turn(
    task: str, variant: Optional[str], seed: dict, state: RelayState, model: str, *,
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print,
) -> Turn:
    """Soạn prompt rồi gọi API NGAY để lấy response -- không còn bước ghi file/chờ người dán.
    Retry (kèm backoff) khi lỗi gọi API hoặc response không parse được JSON hợp lệ; sau
    max_retries lần thất bại, ghi nhận 1 turn "trắng" (parsed_knowledge=None) và relay tiếp tục
    (không dừng cả relay vì 1 lượt lỗi)."""
    prompt = build_turn_prompt(task, variant, seed, state, model)
    round_index = len(state.turns) // len(state.model_order) + 1
    turn_index = len(state.turns) + 1

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
                    f"    --- 300 ký tự ĐẦU ---\n{response[:300]!r}\n"
                    f"    --- 300 ký tự CUỐI ---\n{response[-300:]!r}"
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
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print,
) -> bool:
    """Chạy đủ len(state.model_order) turn. Trả True nếu CẢ round này mọi turn đều vote FINAL."""
    votes = []
    for model in state.model_order:
        turn = run_turn(
            task, variant, seed, state, model,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            log_dir=log_dir, print_fn=print_fn,
        )
        print_fn(f"  round {turn.round_index} [{turn.model}] vote={turn.consensus_vote} "
                  f"parsed={'ok' if turn.parsed_knowledge else 'lỗi'}")
        votes.append(turn.consensus_vote)
    return bool(votes) and all(v == "FINAL" for v in votes)


def run_relay(
    task: str, seed: dict, *, variant: Optional[str] = None,
    model_order: Optional[list[str]] = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print, resume_state: Optional[RelayState] = None,
) -> RelayState:
    state = resume_state or RelayState(task=task, variant=variant, model_order=model_order or DEFAULT_MODEL_ORDER, max_rounds=max_rounds)

    rounds_done = len(state.turns) // len(state.model_order)
    for _ in range(rounds_done, state.max_rounds):
        consensus = run_round(
            task, variant, seed, state,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
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


def run_relay_batched(
    task: str, seed: dict, *, variant: Optional[str] = None,
    model_order: Optional[list[str]] = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
    batch_size: Optional[int] = None, max_workers: int = 4,
    gemini_client=None, gemini_model: str = DEFAULT_GEMINI_MODEL,
    openai_client=None, openai_model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    log_dir: Optional[Path] = None,
    print_fn: Callable[[str], None] = print, resume_states: Optional[list] = None,
) -> list[RelayState]:
    """batch_size=None -> chạy y hệt run_relay() với seed gốc (1 phần tử list trả về) -- giữ
    nguyên hành vi 4 task cũ. batch_size được set -> chia batch, chạy SONG SONG (mỗi batch tự
    tuần tự Gemini<->OpenAI bên trong)."""
    batches = chunk_seed_units(seed, batch_size)
    resume_states = resume_states or [None] * len(batches)

    if len(batches) == 1:
        state = run_relay(
            task, batches[0], variant=variant, model_order=model_order, max_rounds=max_rounds,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            log_dir=log_dir, print_fn=print_fn, resume_state=resume_states[0],
        )
        return [state]

    def _run_one(i: int, batch_seed: dict) -> RelayState:
        batch_log_dir = (log_dir / f"batch_{i:04d}") if log_dir else None
        if batch_log_dir:
            batch_log_dir.mkdir(parents=True, exist_ok=True)
        return run_relay(
            task, batch_seed, variant=variant, model_order=model_order, max_rounds=max_rounds,
            gemini_client=gemini_client, gemini_model=gemini_model,
            openai_client=openai_client, openai_model=openai_model,
            max_retries=max_retries, max_output_tokens=max_output_tokens,
            log_dir=batch_log_dir, print_fn=print_fn, resume_state=resume_states[i],
        )

    states: list = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            i = futures[future]
            states[i] = future.result()
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("run", help="Chạy relay tự động 2 model (Gemini + OpenAI-compatible) qua API.")
    p1.add_argument("--task", required=True, help="Tên task tự do (không giới hạn danh sách cố định) -- dùng để tra INSTRUCTIONS_BY_TASK trong build_debate_seed.py.")
    p1.add_argument("--variant", default=None)
    p1.add_argument("--seed-json", required=True)
    p1.add_argument("--state-output", required=True)
    p1.add_argument("--knowledge-output", required=True)
    p1.add_argument("--gemini-service-account-json", required=True)
    p1.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p1.add_argument("--openai-api-key-file", required=True)
    p1.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL)
    p1.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    p1.add_argument("--model-order", default=",".join(DEFAULT_MODEL_ORDER), help="Danh sách model, phân tách bởi dấu phẩy (giá trị đầu tiên PHẢI hiểu là Gemini nếu là chuỗi \"gemini\", còn lại đều gọi qua OpenAI-compatible client).")
    p1.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p1.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Tăng nếu batch lớn/knowledge dài bị cắt cụt (lỗi 'Không parse được khối JSON hợp lệ').")
    p1.add_argument("--batch-size", type=int, default=None, help="Chia candidate thành nhiều batch (mặc định None = 1 batch duy nhất, đúng hành vi cũ).")
    p1.add_argument("--max-workers", type=int, default=4, help="Số batch chạy song song (chỉ có ý nghĩa khi --batch-size được set).")
    p1.add_argument("--log-dir", default=None, help="Optional: ghi lại prompt/response từng lượt để audit.")
    p1.add_argument("--resume", action="store_true", help="Tiếp tục từ --state-output (hoặc các file batch tương ứng) nếu đã tồn tại.")

    p2 = sub.add_parser("show", help="In lại trạng thái relay đã lưu (không chạy lại).")
    p2.add_argument("--state", required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        with open(args.seed_json, encoding="utf-8") as f:
            seed = json.load(f)

        gemini_client = load_gemini_client(args.gemini_service_account_json)
        openai_client = load_openai_client(Path(args.openai_api_key_file), args.openai_base_url)

        log_dir = Path(args.log_dir) if args.log_dir else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        model_order = args.model_order.split(",")
        state_path = Path(args.state_output)

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
            gemini_client=gemini_client, gemini_model=args.gemini_model,
            openai_client=openai_client, openai_model=args.openai_model,
            max_retries=args.max_retries, max_output_tokens=args.max_output_tokens,
            log_dir=log_dir, resume_states=resume_states,
        )
        save_batch_states(states, state_path)

        if len(states) == 1:
            export_knowledge_graph(states[0], args.task, args.variant, Path(args.knowledge_output))
        else:
            export_merged_knowledge_graph(
                states, args.task, args.variant, Path(args.knowledge_output),
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
