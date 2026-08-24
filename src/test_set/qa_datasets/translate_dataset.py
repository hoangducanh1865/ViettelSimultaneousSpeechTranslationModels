"""Generic EN->VI batch translator for JSON text datasets (Gemini/Vertex AI),
used for MusicBench (caption fields) and MusicQA (question/answer pairs).

Core design:
- Operates on "translation units" (TranslationUnit: a sample index + a dict
  of {field_name: source_text}) so it doesn't need to know each dataset's
  record shape. Callers (notebook cells) extract units from raw records and
  merge translated fields back -- keeps this module reusable for future
  datasets with yet another shape.
- A unit's fields are always translated TOGETHER in the same Gemini call, in
  the same JSON object -- this is what keeps a question/answer pair (or any
  other multi-field group) semantically consistent after translation: the
  model sees both at once and translates them as one coherent exchange,
  rather than translating each field blind to the other.
- Batching: several units per Gemini call (--batch-size), reducing request
  count/cost. Batches run concurrently (--max-workers); ThreadPoolExecutor,
  matching the concurrency pattern already used in rover.py.
- Validation, never silent corruption: each unit's translated fields are
  checked for (a) actually being Vietnamese (langdetect) and (b) a sane
  length ratio vs the source (catches empty/truncated/rambling output). A
  unit that fails either check gets retried alone (single-unit call, more
  room for the model to get it right without other units' noise); if it
  still fails, the whole sample is marked failed and excluded from the
  clean output -- never partially translated, never silently kept as
  mistranslated/still-English text.
- Resumable: writes one JSON line per sample to --output as it completes,
  and skips sample indices already present there on a re-run.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3


def load_json_or_jsonl(path):
    """Some HF dataset exports use a `.json` extension but are actually
    JSON Lines (one object per line) rather than a single JSON array --
    MusicBench_train.json is one such case. Try a normal whole-file parse
    first, fall back to line-by-line JSONL on failure."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


MIN_LEN_RATIO = 0.3
MAX_LEN_RATIO = 3.5

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class TranslationUnit:
    sample_index: int
    fields: dict[str, str]  # field_name -> source text (all translated together)


@dataclass
class TranslationResult:
    sample_index: int
    ok: bool
    translated_fields: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    mode: str = "batched"  # "batched" | "fallback_single"


SYSTEM_PROMPT = """\
Bạn là công cụ dịch thuật tiếng Anh sang tiếng Việt cho dataset caption/hỏi-đáp \
về âm nhạc. Dịch chính xác, tự nhiên, giữ đúng ý nghĩa; thuật ngữ âm nhạc (tên \
nhạc cụ, thể loại, kỹ thuật chơi) dịch theo cách người Việt trong ngành nhạc \
thường dùng, không dịch máy móc từng chữ. Tên riêng (tên bài hát, nghệ sĩ, tên \
file) giữ nguyên không dịch.

Đầu vào là một mảng JSON, mỗi phần tử có "id" và một hoặc nhiều field text cần \
dịch. Nếu một phần tử có nhiều field (ví dụ "question" và "answer"), các field \
đó thuộc cùng một đoạn hội thoại/ngữ cảnh -- PHẢI dịch sao cho chúng vẫn khớp \
nghĩa với nhau sau khi dịch (không được để câu hỏi hỏi một đằng, câu trả lời \
dịch thành nói chuyện khác).

Trả về ĐÚNG một mảng JSON, cùng số phần tử, cùng "id", cùng tên field như đầu \
vào -- chỉ thay giá trị text bằng bản dịch tiếng Việt. Không thêm field, không \
bỏ field, không thêm giải thích, không dùng markdown code fence. Field rỗng thì \
giữ nguyên rỗng.
"""


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


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _call_gemini_json(client, model: str, user_content: str, *, max_output_tokens: int) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        ),
    )
    return (response.text or "").strip()


def _call_with_retry(client, model: str, user_content: str, *, max_output_tokens: int, max_retries: int) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return _call_gemini_json(client, model, user_content, max_output_tokens=max_output_tokens)
        except Exception as e:  # noqa: BLE001 -- transient API errors (429/5xx/timeout), retry all
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error!r}")


def is_vietnamese_text(text: str) -> bool:
    """True if `text` looks like Vietnamese, used to catch translations that
    silently came back still in English (empty text is vacuously fine --
    nothing to translate, not a failure)."""
    if not text or not text.strip():
        return True
    try:
        from langdetect import detect, DetectorFactory

        DetectorFactory.seed = 0  # deterministic
        return detect(text) == "vi"
    except Exception:
        # langdetect unavailable or failed to classify (very short/ambiguous
        # text) -- fall back to a crude but dependency-free signal: Vietnamese
        # text almost always contains at least one diacritic character for
        # anything longer than a couple of words.
        vietnamese_chars = re.findall(
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
            r"ùúụủũưừứựửữỳýỵỷỹđ]", text, flags=re.IGNORECASE,
        )
        word_count = len(text.split())
        return word_count <= 2 or len(vietnamese_chars) > 0


def is_reasonable_length(source: str, target: str) -> bool:
    """Loose sanity check -- catches empty/truncated or wildly rambling
    output. Not a fidelity check (that's what QA-pair joint translation and
    is_vietnamese_text are for)."""
    if not source or not source.strip():
        return True
    if not target or not target.strip():
        return False
    ratio = len(target) / len(source)
    return MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO


def validate_unit(unit: TranslationUnit, translated: dict[str, str]) -> Optional[str]:
    """Returns None if `translated` passes validation, else a reason string."""
    for field_name, source_text in unit.fields.items():
        if field_name not in translated:
            return f"missing field '{field_name}' in response"
        target_text = translated[field_name]
        if not is_reasonable_length(source_text, target_text):
            return f"field '{field_name}': length ratio out of range"
        if not is_vietnamese_text(target_text):
            return f"field '{field_name}': output doesn't look like Vietnamese"
    return None


def _build_prompt(units: list[TranslationUnit]) -> str:
    payload = [{"id": u.sample_index, **u.fields} for u in units]
    return json.dumps(payload, ensure_ascii=False)


def _parse_response(response_text: str, units: list[TranslationUnit]) -> Optional[dict[int, dict[str, str]]]:
    try:
        data = json.loads(_strip_fences(response_text))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None

    by_id: dict[int, dict[str, str]] = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            return None
        by_id[item["id"]] = {k: v for k, v in item.items() if k != "id"}

    expected_ids = {u.sample_index for u in units}
    if set(by_id.keys()) != expected_ids:
        return None
    return by_id


def translate_unit_single(client, model: str, unit: TranslationUnit, *, max_retries: int) -> TranslationResult:
    """Retries up to max_retries full attempts -- each one a fresh Gemini
    call -- as long as the *result* keeps failing validation (not just on
    API errors, which _call_gemini_json already retries internally). Gives
    the model repeated independent chances to get this one stubborn unit
    right before giving up on it."""
    prompt = _build_prompt([unit])
    last_error = "unknown error"

    for attempt in range(max_retries):
        try:
            response_text = _call_gemini_json(client, model, prompt, max_output_tokens=2048)
        except Exception as e:
            last_error = repr(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

        parsed = _parse_response(response_text, [unit])
        if parsed is None:
            last_error = "malformed JSON response"
            continue

        translated = parsed[unit.sample_index]
        reason = validate_unit(unit, translated)
        if reason is None:
            return TranslationResult(unit.sample_index, ok=True, translated_fields=translated, mode="fallback_single")
        last_error = reason

    return TranslationResult(unit.sample_index, ok=False, error=last_error, mode="fallback_single")


def translate_batch(
    client, model: str, units: list[TranslationUnit], *, max_output_tokens_per_unit: int, max_retries: int,
) -> list[TranslationResult]:
    prompt = _build_prompt(units)
    max_output_tokens = max(2048, max_output_tokens_per_unit * len(units))

    try:
        response_text = _call_with_retry(
            client, model, prompt, max_output_tokens=max_output_tokens, max_retries=max_retries
        )
        parsed = _parse_response(response_text, units)
    except Exception:
        parsed = None

    if parsed is None:
        # Whole batch came back malformed/mismatched -- no reliable way to
        # tell which units are actually fine, so retry every one alone
        # rather than guessing.
        return [translate_unit_single(client, model, u, max_retries=max_retries) for u in units]

    results = []
    for unit in units:
        translated = parsed[unit.sample_index]
        reason = validate_unit(unit, translated)
        if reason is None:
            results.append(TranslationResult(unit.sample_index, ok=True, translated_fields=translated))
        else:
            # Only this unit failed validation -- retry it alone, keep the
            # rest of the batch's (valid) results as-is.
            results.append(translate_unit_single(client, model, unit, max_retries=max_retries))
    return results


def load_processed_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    indices = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(json.loads(line)["sample_index"])
    return indices


def translate_units(
    client,
    model: str,
    units: list[TranslationUnit],
    output_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_output_tokens_per_unit: int = 512,
    force: bool = False,
) -> dict:
    """Translate `units`, appending one JSON line per completed sample to
    `output_path` as results come in (resumable: already-processed sample
    indices are skipped on a re-run unless force=True).

    Each output line: {"sample_index": int, "ok": bool,
    "translated_fields": {...}, "error": str|None, "mode": str}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if force else load_processed_indices(output_path)
    pending = [u for u in units if u.sample_index not in done]

    print(f"{len(units) - len(pending)}/{len(units)} sample(s) already translated, {len(pending)} to do.")

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    stats = {"kept": 0, "failed": 0}
    lock = threading.Lock()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    translate_batch, client, model, batch,
                    max_output_tokens_per_unit=max_output_tokens_per_unit, max_retries=max_retries,
                ): batch
                for batch in batches
            }
            n_done_batches = 0
            for future in as_completed(futures):
                results = future.result()
                with lock:
                    for r in results:
                        out_f.write(json.dumps({
                            "sample_index": r.sample_index,
                            "ok": r.ok,
                            "translated_fields": r.translated_fields,
                            "error": r.error,
                            "mode": r.mode,
                        }, ensure_ascii=False) + "\n")
                        stats["kept" if r.ok else "failed"] += 1
                    out_f.flush()
                    n_done_batches += 1
                    print(f"  batch {n_done_batches}/{len(batches)} done "
                          f"(kept={stats['kept']}, failed={stats['failed']})")

    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-account-json", required=True)
    parser.add_argument("--units-file", required=True,
                         help="JSONL file, one {\"sample_index\": int, \"fields\": {...}} per line "
                         "(produced by the caller's dataset-specific extraction step).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    units = []
    with open(args.units_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                units.append(TranslationUnit(sample_index=d["sample_index"], fields=d["fields"]))
    if args.limit is not None:
        units = units[: args.limit]

    client = load_gemini_client(args.service_account_json)
    stats = translate_units(
        client, args.model, units, Path(args.output),
        batch_size=args.batch_size, max_workers=args.max_workers,
        max_retries=args.max_retries, force=args.force,
    )
    print("Totals:", stats)


if __name__ == "__main__":
    main()
