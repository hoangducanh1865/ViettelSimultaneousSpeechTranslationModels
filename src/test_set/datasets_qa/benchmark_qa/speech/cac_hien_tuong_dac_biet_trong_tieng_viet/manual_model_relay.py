"""Drives a manual copy-paste relay between 3 free web-tier chat models (Gemini web, ChatGPT web,
Claude web) to build ONE task's knowledge-graph JSON (see knowledge_graph.py for the schema).

NO paid API calls, NO OpenAI/Anthropic/Google SDK usage, NO network calls of any kind -- this
module only builds prompt text, and hands it off via a pair of relay FILES rather than printing
it straight to the terminal (a turn's prompt/response can be many KB -- unwieldy to read/paste
through a terminal buffer):
  - --prompt-file: this script WRITES the current turn's prompt here. The human opens this file,
    copies its ENTIRE content, and pastes it into the model's web chat (Gemini/ChatGPT/Claude).
  - --response-file: the human pastes the model's reply into this file and saves it. The script
    only asks a short "đã dán xong chưa?" confirmation via input() (never the reply text itself),
    then reads the file.
This is a deliberate cost-control decision: knowledge-building happens rarely (once per task, or
whenever a human wants to refresh it), while the actual bulk question generation downstream still
uses the cheap Gemini API (gemini-3.1-flash-lite) unchanged.

MUST run in a REAL LOCAL TERMINAL: input() needs interactive stdin, which Colab's `!python ...`
shell-out cannot provide -- this can never be a notebook cell.

Stopping condition: a round (one turn per model in --model-order) ends the relay the moment
EVERY model's turn that round contains the literal marker "CONSENSUS: FINAL"; otherwise the
relay continues until --max-rounds is exhausted, at which point the LATEST successfully-parsed
knowledge JSON is used regardless of consensus ("best available" fallback) -- this matches the
"cả 3 model đồng ý + giới hạn vòng" stopping rule the user confirmed.

Usage:
    python manual_model_relay.py run --task tu_lay --variant toan_bo \\
        --seed-json debate_seed_tu_lay_toan_bo.json \\
        --state-output relay_state_tu_lay_toan_bo.json \\
        --knowledge-output knowledge_tu_lay_toan_bo.json \\
        --prompt-file relay_prompt_tu_lay_toan_bo.txt \\
        --response-file relay_response_tu_lay_toan_bo.txt

    python manual_model_relay.py show --state relay_state_tu_lay_toan_bo.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

DEFAULT_MODEL_ORDER = ["gemini", "chatgpt", "claude"]
DEFAULT_MAX_ROUNDS = 5

_CONSENSUS_MARKER_RE = re.compile(r"CONSENSUS:\s*(FINAL|CONTINUE)", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_SKIP_SENTINEL = "SKIP"


@dataclass
class Turn:
    round_index: int
    turn_index: int
    model: str
    prompt: str
    response: str
    parsed_knowledge: Optional[dict]
    consensus_vote: Optional[str]  # "FINAL" | "CONTINUE" | None (marker missing in paste)


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
    cuối luôn là bản họ muốn giữ lại)."""
    matches = _JSON_FENCE_RE.findall(response_text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None


def extract_consensus_vote(response_text: str) -> Optional[str]:
    m = _CONSENSUS_MARKER_RE.search(response_text)
    return m.group(1).upper() if m else None


def build_turn_prompt(task: str, variant: Optional[str], seed: dict, state: RelayState, model: str) -> str:
    """Turn 1 (state.turns rỗng): instructions + schema + candidate_words + sample transcript
    chưa khớp từ nào, lấy nguyên từ seed. Turn >1: tóm tắt NGẮN turn ngay trước (model nào, họ
    vote gì) + TOÀN BỘ knowledge JSON hiện tại (mỗi model luôn sửa trên bản đầy đủ mới nhất,
    không phải diff -- giữ prompt tự-chứa, không cần người dùng nhớ lại lịch sử) -- cùng yêu cầu
    CONSENSUS ở cuối."""
    header = f"[Relay tri thức -- task={task}" + (f", variant={variant}" if variant else "") + f"] Lượt cho model: {model.upper()}\n"

    if not state.turns:
        body = (
            f"{seed['instructions']}\n\n"
            f"SCHEMA đích (trả về ĐÚNG dạng này):\n{seed['schema_spec']}\n\n"
            f"DANH SÁCH TỪ CANDIDATE (đã gộp từ CSV hiện có + web-research + quét corpus, "
            f"kèm bằng chứng thật nếu có):\n{json.dumps(seed['candidate_words'], ensure_ascii=False, indent=2)}\n\n"
            f"MỘT SỐ TRANSCRIPT CHƯA KHỚP TỪ NÀO (tham khảo để đề xuất từ MỚI nếu bạn thấy phù hợp):\n"
            f"{json.dumps(seed['unmatched_transcript_sample'], ensure_ascii=False, indent=2)}"
        )
    else:
        prev = state.turns[-1]
        body = (
            f"Model trước ({prev.model}) vừa sửa knowledge graph và vote "
            f"CONSENSUS: {prev.consensus_vote or 'CONTINUE (không rõ, thiếu marker)'}.\n\n"
            f"{seed['instructions']}\n\n"
            f"SCHEMA đích:\n{seed['schema_spec']}\n\n"
            f"KNOWLEDGE GRAPH HIỆN TẠI (hãy rà soát/sửa/thêm/bớt, rồi trả lại TOÀN BỘ, không phải diff):\n"
            f"{json.dumps(state.latest_knowledge, ensure_ascii=False, indent=2)}"
        )

    return header + "\n" + body


def run_turn(
    task: str, variant: Optional[str], seed: dict, state: RelayState, model: str, *,
    prompt_path: Path, response_path: Path,
    input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
) -> Turn:
    """Ghi prompt vào prompt_path (KHÔNG in ra terminal -- prompt/response có thể dài hàng chục
    KB, bất tiện copy/paste qua buffer terminal). Xóa response_path CŨ trước (tránh đọc nhầm câu
    trả lời của lượt trước nếu người dùng quên ghi đè). input_fn() chỉ hỏi xác nhận NGẮN đã dán
    xong chưa, không nhận trực tiếp nội dung dài qua stdin."""
    prompt = build_turn_prompt(task, variant, seed, state, model)
    prompt_path.write_text(prompt, encoding="utf-8")
    if response_path.exists():
        response_path.unlink()

    round_index = len(state.turns) // len(state.model_order) + 1
    turn_index = len(state.turns) + 1

    print_fn(f"\n>>> Round {round_index}, lượt của {model.upper()}:")
    print_fn(f"    1. Mở {prompt_path}, copy TOÀN BỘ nội dung, dán vào {model.upper()} (web).")
    print_fn(f"    2. Dán câu trả lời của {model.upper()} vào {response_path} rồi lưu file lại.")

    while True:
        answer = input_fn(f"Đã dán xong câu trả lời vào {response_path.name}? (Enter để đọc tiếp, gõ {_SKIP_SENTINEL} để bỏ qua lượt này): ")
        if answer.strip().upper() == _SKIP_SENTINEL:
            turn = Turn(round_index, turn_index, model, prompt, "", None, None)
            state.turns.append(turn)
            return turn

        if not response_path.exists() or not response_path.read_text(encoding="utf-8").strip():
            print_fn(f"[LỖI] {response_path} vẫn trống -- dán câu trả lời vào file rồi Enter lại, hoặc gõ {_SKIP_SENTINEL}.")
            continue

        response = response_path.read_text(encoding="utf-8")
        parsed = extract_knowledge_json(response)
        if parsed is None:
            print_fn(f"[LỖI] Không tìm/parse được khối ```json``` hợp lệ trong {response_path} -- sửa lại file rồi Enter, hoặc gõ {_SKIP_SENTINEL}.")
            continue

        vote = extract_consensus_vote(response)
        turn = Turn(round_index, turn_index, model, prompt, response, parsed, vote)
        state.turns.append(turn)
        state.latest_knowledge = parsed
        return turn


def run_round(
    task: str, variant: Optional[str], seed: dict, state: RelayState, *,
    prompt_path: Path, response_path: Path,
    input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
) -> bool:
    """Chạy đủ len(state.model_order) turn. Trả True nếu CẢ round này mọi turn đều vote FINAL."""
    votes = []
    for model in state.model_order:
        turn = run_turn(
            task, variant, seed, state, model,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn, print_fn=print_fn,
        )
        votes.append(turn.consensus_vote)
    return all(v == "FINAL" for v in votes)


def run_relay(
    task: str, seed: dict, *, variant: Optional[str] = None,
    model_order: list[str] = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
    prompt_path: Path, response_path: Path,
    input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
    resume_state: Optional[RelayState] = None,
) -> RelayState:
    state = resume_state or RelayState(task=task, variant=variant, model_order=model_order or DEFAULT_MODEL_ORDER, max_rounds=max_rounds)

    rounds_done = len(state.turns) // len(state.model_order)
    for _ in range(rounds_done, state.max_rounds):
        consensus = run_round(
            task, variant, seed, state,
            prompt_path=prompt_path, response_path=response_path, input_fn=input_fn, print_fn=print_fn,
        )
        if consensus:
            state.stopped_reason = "consensus"
            print_fn(f"\n>>> Cả {len(state.model_order)} model đồng thuận CONSENSUS: FINAL -- dừng relay.")
            return state

    state.stopped_reason = "max_rounds"
    print_fn(f"\n>>> Đã hết {state.max_rounds} vòng mà chưa đồng thuận -- dùng bản tri thức mới nhất hiện có.")
    return state


def _turn_to_dict(t: Turn) -> dict:
    return asdict(t)


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
    """Chỉ ghi latest_knowledge + relay_meta -- đây là artifact THẬT mà các pipeline khác tiêu
    thụ qua --knowledge-json, KHÔNG phải toàn bộ turn log (đó là việc của save_state, dùng cho
    audit/resume)."""
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("run", help="Chạy relay 3 model tương tác (input() nhiều lượt).")
    p1.add_argument("--task", required=True, choices=["han_viet", "tu_muon", "tu_lay", "phuong_ngu"])
    p1.add_argument("--variant", default=None, choices=["toan_bo", "van", "chung"])
    p1.add_argument("--seed-json", required=True)
    p1.add_argument("--state-output", required=True)
    p1.add_argument("--knowledge-output", required=True)
    p1.add_argument("--prompt-file", default=None, help="File trung chuyển để COPY prompt ra web (mặc định: relay_prompt_<task>[_<variant>].txt cạnh --state-output).")
    p1.add_argument("--response-file", default=None, help="File trung chuyển để PASTE câu trả lời model vào (mặc định: relay_response_<task>[_<variant>].txt cạnh --state-output).")
    p1.add_argument("--model-order", default=",".join(DEFAULT_MODEL_ORDER), help="Danh sách model, phân tách bởi dấu phẩy.")
    p1.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p1.add_argument("--resume", action="store_true", help="Tiếp tục từ --state-output nếu đã tồn tại.")

    p2 = sub.add_parser("show", help="In lại trạng thái relay đã lưu (không chạy lại).")
    p2.add_argument("--state", required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        with open(args.seed_json, encoding="utf-8") as f:
            seed = json.load(f)

        resume_state = None
        state_path = Path(args.state_output)
        if args.resume and state_path.exists():
            resume_state = load_state(state_path)
            print(f"Resume từ {state_path}: đã có {len(resume_state.turns)} turn.")

        suffix = f"_{args.task}" + (f"_{args.variant}" if args.variant else "")
        prompt_path = Path(args.prompt_file) if args.prompt_file else state_path.with_name(f"relay_prompt{suffix}.txt")
        response_path = Path(args.response_file) if args.response_file else state_path.with_name(f"relay_response{suffix}.txt")

        state = run_relay(
            args.task, seed, variant=args.variant, model_order=args.model_order.split(","),
            max_rounds=args.max_rounds, prompt_path=prompt_path, response_path=response_path,
            resume_state=resume_state,
        )
        save_state(state, state_path)
        export_knowledge_graph(state, args.task, args.variant, Path(args.knowledge_output))

    elif args.command == "show":
        state = load_state(Path(args.state))
        print(f"task={state.task} variant={state.variant} stopped_reason={state.stopped_reason} "
              f"turns={len(state.turns)}/{state.max_rounds * len(state.model_order)}")
        for t in state.turns:
            print(f"  round {t.round_index} turn {t.turn_index} [{t.model}] vote={t.consensus_vote} "
                  f"parsed={'ok' if t.parsed_knowledge else 'SKIP/lỗi'}")


if __name__ == "__main__":
    main()
