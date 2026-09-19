#!/usr/bin/env python3
"""Lọc chất lượng câu hỏi trắc nghiệm benchmark QA bằng 2 API model MẠNH, dùng CHUNG cho mọi task
(Hán Việt, Phương ngữ, Từ mượn, Từ láy, Code-switching).

Một lượt = 1 provider (Gemini hoặc OpenAI-compatible). Luồng chuẩn chạy 2 lượt liên tiếp:
    filter-qa --provider gemini  ...   (model mặc định: gemini-3.1-pro-preview)
    filter-qa --provider openai  ...   (model mặc định: cx/gpt-5.6-luna, input = output lượt Gemini)

Điểm khác biệt so với code_switching_qa_pipeline.filter-questions (bản cũ, chỉ dùng cho
Code-switching):
  - Prompt lọc CHI TIẾT theo từng task (tiêu chí chung + tiêu chí đặc thù của task).
  - Gửi kèm CONTEXT theo task (base_word/tone_register/region_or_ethnic_group/level/...) để model
    có đủ dữ kiện xác minh (vd Tone Harmony của từ láy, vùng/dân tộc của phương ngữ).
  - Khi gọi API lỗi: GIỮ LẠI record (mặc định an toàn), ghi rõ reason.

Input : file pre-final JSONL, mỗi record tối thiểu có {id, question, choices, answer}.
Output: file kept JSONL (thêm field `filter_reason`) + file rules JSON (`criteria_summaries`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))  # auto_model_relay.py, env_paths.py cùng thư mục

import auto_model_relay  # noqa: E402
import env_paths  # noqa: E402

DEFAULT_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 8192

# Model MẠNH mặc định cho bước lọc (lấy từ hằng số dùng chung của auto_model_relay).
DEFAULT_GEMINI_FILTER_MODEL = auto_model_relay.DEFAULT_GEMINI_MODEL
DEFAULT_OPENAI_FILTER_MODEL = auto_model_relay.DEFAULT_OPENAI_MODEL

# Field ngữ cảnh gửi kèm cho model, tuỳ task (chỉ gửi field nào record thực sự có).
CONTEXT_FIELDS: dict[str, list[str]] = {
    "han_viet": ["level", "max_level", "target_word", "historical_fact"],
    "tu_muon": ["level", "max_level", "historical_fact"],
    "tu_lay": ["level", "max_level", "base_word", "tone_register", "question_type", "cultural_fact"],
    "phuong_ngu": ["region_or_ethnic_group", "question_type"],
    "code_switching": ["question_type", "target_terms"],
}

# =============================================================================================
# Prompt lọc chi tiết theo task
# =============================================================================================

_BASE_CRITERIA = """Bạn là giám khảo chất lượng cho câu hỏi trắc nghiệm 4 lựa chọn của một benchmark
nghe-hiểu tiếng Việt. Chấm từng câu theo các tiêu chí sau, KHÔNG nương tay nhưng cũng KHÔNG loại
oan (nếu chỉ nghi ngờ mà không chứng minh được lỗi, hãy GIỮ LẠI).

TIÊU CHÍ CHUNG (mọi task):
1. Tự nhiên: câu hỏi đọc lên nghe tự nhiên, đúng ngữ pháp, không gượng ép/máy móc/dịch máy.
2. Đa dạng: không lặp y hệt cấu trúc/cách hỏi của các câu khác trong cùng batch.
3. Độ khó thực chất: đáp án KHÔNG thể suy ra nếu không nghe/hiểu nội dung; loại câu mà đáp án quá
   hiển nhiên, hỏi kiến thức phổ thông không cần audio, hoặc đoán mò là ra.
4. Nhiễu hợp lý: cả 3 phương án sai đều hợp lý, cùng loại/phạm trù với đáp án; KHÔNG có phương án
   vô lý, lạc đề, hoặc dễ loại trừ ngay; KHÔNG có phương án trùng nghĩa với đáp án đúng.
5. Đáp án hợp lệ: "answer" PHẢI nằm trong "choices"; nếu không, loại ngay.
6. Chỉ có MỘT đáp án đúng: nếu có từ 2 phương án trở lên cùng đúng (dù diễn đạt khác), loại.
7. Chống lộ đáp án: câu hỏi KHÔNG được chứa nguyên văn đáp án đúng hoặc từ khoá mục tiêu (nếu có
   context cung cấp).
8. Chống "ăn may": loại câu mà đáp án đúng là phương án dài nhất/rõ ràng nhất/khác loại rõ rệt so
   với 3 phương án còn lại (dấu hiệu đoán được không cần hiểu nội dung).
"""

_OUTPUT_FORMAT = """Với MỖI câu hỏi, quyết định "keep": true/false và "reason" (lý do ngắn gọn; nếu loại
phải nêu RÕ tiêu chí số mấy không đạt).

Input: JSON array các object (mỗi object tối thiểu có "id", "question", "choices", "answer"; có
thể kèm field ngữ cảnh đặc thù của task, hãy DÙNG chúng để chấm chính xác hơn).

Output: CHỈ trả về JSON object:
{"items": [{"id": <id đầu vào>, "keep": true|false, "reason": str}, ...],
 "criteria_summary": str (tóm tắt NGẮN GỌN ngưỡng/tiêu chí bạn ĐÃ ÁP DỤNG thực tế cho batch này)}
-- không giải thích thêm, không markdown fence.
"""

_TASK_SPECIFIC: dict[str, str] = {
    "han_viet": """ĐẶC THÙ TASK HÁN VIỆT:
- Câu hỏi KHÔNG được nhắc thẳng từ Hán-Việt mục tiêu (người nghe phải tự nhận ra qua audio); nếu
  câu hỏi (hoặc lựa chọn) chứa chính "target_word" thì LOẠI.
- Xác minh đúng cấp độ (context "level"):
  * Level 1 (1-hop): hỏi nghĩa đen; nhiễu phải là nghĩa thật khác, không đồng nghĩa.
  * Level 2 (2-hop): hỏi phạm trù/lĩnh vực SUY RA TỪ NGỮ CẢNH transcript, không phải tra nghĩa;
    nếu câu chỉ đòi nghĩa từ thì HẠ/LOẠI.
  * Level 3 (3-hop): đáp án phải gắn với "historical_fact" THẬT đã cho; loại nếu fact mơ hồ, bịa,
    hoặc sự kiện chỉ liên hệ gượng ép.
- Ưu tiên GIỮ câu đòi reasoning sâu (level 2/3) và câu có nhiễu cùng phạm trù.
""",
    "tu_muon": """ĐẶC THÙ TASK TỪ MƯỢN (loanword đã Việt hoá):
- Câu hỏi KHÔNG được nhắc thẳng từ mượn mục tiêu; nếu chứa chính đáp án/từ mục tiêu thì LOẠI.
- Xác minh đúng cấp độ (context "level"):
  * Level 1: hỏi 1 khía cạnh thật (ngôn ngữ nguồn gốc / từ gốc / ý nghĩa); đáp án phải khớp dữ kiện.
  * Level 2: hỏi "nhóm lĩnh vực sử dụng"; nhiễu phải là các lĩnh vực THẬT khác, không trùng nghĩa.
  * Level 3: bối cảnh lịch sử-văn hoá du nhập THẬT ("historical_fact"); loại nếu bịa/mơ hồ.
- Ưu tiên GIỮ câu có nhiễu CÙNG ngôn ngữ nguồn + CÙNG lĩnh vực (khó, dễ nhầm), và câu level 3 xác
  thực được bằng kiến thức lịch sử phổ biến.
""",
    "tu_lay": """ĐẶC THÙ TASK TỪ LÁY:
- Level 1/2/3 (context "level"): level 1 = cấu tạo/ngữ âm (loại từ láy, phần vần, từ loại) hoặc
  nghĩa/sắc thái; level 2 = ngữ cảnh/phong cách dùng (dựa trên "Sắc thái biểu đạt"); level 3 = dẫn
  chứng ca dao/tục ngữ/thành ngữ/tác phẩm văn học THẬT ("cultural_fact"). Loại level 3 nếu dẫn
  chứng bịa/sai/không thật sự dùng chính từ láy đó.
- Câu hỏi cloze/Tone Harmony (context "question_type" == "cloze-tone-harmony", CHỈ variant toàn bộ)
  là cơ chế QUAN TRỌNG NHẤT, chấm RẤT nghiêm:
  * Đáp án đúng PHẢI là TỪ LÁY TOÀN BỘ HỢP LỆ của "base_word" trong transcript, tuân thủ quy tắc
    HÒA ÂM VỰC (Tone Harmony): âm vực CAO (thanh ngang, sắc, hỏi) chỉ ghép với âm vực CAO; âm vực
    THẤP (thanh huyền, nặng, ngã) chỉ ghép với âm vực THẤP; TUYỆT ĐỐI KHÔNG bắc cầu âm vực.
    Ví dụ đúng: "xanh" (ngang, cao) -> "xanh xao"; "đẹp" (nặng, thấp) -> "đẹp đẽ"; "nhẹ" (nặng,
    thấp) -> "nhè nhẹ". Ví dụ SAI (phải loại): "nhẹ nhẹ" (sai quy tắc láy, chỉ lặp máy móc).
  * LOẠI nếu đáp án chỉ là lặp lại y hệt từ gốc ("xanh xanh" khi sai quy tắc, "nhẹ nhẹ", ...), hoặc
    không phải từ láy toàn bộ thật của "base_word".
  * 3 nhiễu phải là TỪ LÁY THẬT khác, ưu tiên CÙNG âm vực với "base_word" (khó nhất), KHÔNG trùng
    nghĩa đáp án; loại nếu nhiễu là từ bịa/không tồn tại.
  * Hướng câu hỏi (tăng/giảm mức độ) phải khớp với sắc thái của đáp án; loại nếu hỏi "tăng" mà đáp
    án lại mang nghĩa "giảm" (hoặc ngược lại).
- Ưu tiên GIỮ câu cloze đúng Tone Harmony và câu level 1 đòi phân tích cấu trúc ngữ âm.
""",
    "phuong_ngu": """ĐẶC THÙ TASK PHƯƠNG NGỮ:
- Câu hỏi BẮT BUỘC phải hỏi về VÙNG MIỀN hoặc DÂN TỘC gắn với từ/cụm từ phương ngữ; nếu câu hỏi
  lệch sang nghĩa của từ, cách viết, hay nội dung khác thì LOẠI.
- Đáp án PHẢI khớp với "region_or_ethnic_group" đã cho trong context; nếu lệch thì LOẠI.
- 3 nhiễu PHẢI là tên vùng miền/dân tộc THẬT khác của Việt Nam, khác nhau, khác đáp án; LOẠI nếu
  có nhiễu bịa tên vùng/dân tộc không tồn tại, hoặc trùng đáp án.
- Câu hỏi KHÔNG được nhắc thẳng từ phương ngữ mục tiêu; loại nếu lộ từ khoá.
- Ưu tiên GIỮ câu hỏi buộc suy luận vùng CỤ THỂ (vd Nghệ Tĩnh, Huế, Nam Trung Bộ...) thay vì chỉ
  Bắc/Trung/Nam, và câu có nhiễu là các vùng HAY BỊ NHẦM với đáp án.
""",
    "code_switching": """ĐẶC THÙ TASK CODE-SWITCHING:
- Câu hỏi KHÔNG được lộ "target_terms" (các thuật ngữ code-switching mục tiêu); nếu chứa thẳng
  thuật ngữ đang hỏi thì LOẠI.
- Kiểm tra đúng loại câu hỏi (context "question_type"):
  * A đếm/từ loại: số đếm hoặc từ loại phải khớp dữ kiện thật.
  * B tác động ngữ nghĩa: đáp án phải là tác động hợp lý của thuật ngữ trong ngữ cảnh.
  * C tri thức nền: dữ kiện phải THẬT, không bịa.
  * D Việt hoá: phải có phương án Việt hoá đúng, các phương án khác không đồng nghĩa.
  * E rủi ro ASR: tình huống rủi ro phải hợp lý với thuật ngữ.
  * F quan hệ đa-thuật-ngữ: quan hệ giữa nhiều thuật ngữ phải đúng và rõ ràng.
- Nhiễu không được trùng nghĩa đáp án và không được lộ "target_terms".
""",
}


def build_filter_system_prompt(task: str) -> str:
    if task not in _TASK_SPECIFIC:
        raise ValueError(f"Task không hợp lệ: {task!r} (phải là 1 trong {sorted(_TASK_SPECIFIC)}).")
    return _BASE_CRITERIA + "\n" + _TASK_SPECIFIC[task] + "\n" + _OUTPUT_FORMAT


def build_payload_item(r: dict, context_fields: list[str]) -> dict:
    item = {"id": r["id"], "question": r["question"], "choices": r["choices"], "answer": r["answer"]}
    for f in context_fields:
        if f in r and r[f] not in (None, "", [], {}):
            item[f] = r[f]
    return item


# =============================================================================================
# Gọi provider
# =============================================================================================

def make_provider_call(provider: str, client, model: str):
    def _call(payload_json: str, system_prompt: str) -> str:
        return auto_model_relay.call_model(
            provider, payload_json, gemini_client=client, gemini_model=model,
            openai_client=client, openai_model=model,
            system_instruction=system_prompt, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS,
        )

    return _call


def _filter_batch(provider_call, batch: list[dict], system_prompt: str, context_fields: list[str],
                  max_retries: int):
    payload = [build_payload_item(r, context_fields) for r in batch]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = provider_call(json.dumps(payload, ensure_ascii=False), system_prompt)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            items = {item["id"]: item for item in parsed.get("items", []) if item.get("id") in ids_sent}
            missing = ids_sent - set(items)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return items, parsed.get("criteria_summary", "")
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI lọc câu hỏi] batch {len(batch)} câu: {last_error!r} -- GIỮ LẠI toàn bộ batch (mặc định an toàn).")
    return {r["id"]: {"keep": True, "reason": "Lỗi gọi API, mặc định giữ lại."} for r in batch}, ""


def filter_questions(provider_call, records: list[dict], kept_output: Path, rules_output: Path,
                     *, system_prompt: str, context_fields: list[str],
                     batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
                     max_retries: int = DEFAULT_MAX_RETRIES) -> None:
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    all_items: dict = {}
    all_summaries: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_filter_batch, provider_call, b, system_prompt, context_fields, max_retries): b
                   for b in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="lọc câu hỏi"):
            items, summary = future.result()
            all_items.update(items)
            if summary:
                all_summaries.append(summary)

    n_kept = 0
    with open(kept_output, "w", encoding="utf-8") as f:
        for r in records:
            verdict = all_items.get(r["id"], {"keep": True, "reason": "Không có verdict (lỗi) -- mặc định giữ lại."})
            if not verdict.get("keep"):
                continue
            out_r = dict(r)
            out_r["filter_reason"] = verdict.get("reason", "")
            f.write(json.dumps(out_r, ensure_ascii=False) + "\n")
            n_kept += 1

    with open(rules_output, "w", encoding="utf-8") as f:
        json.dump({"criteria_summaries": all_summaries}, f, ensure_ascii=False, indent=2)

    print(f"Đã lọc: giữ {n_kept}/{len(records)} câu hỏi vào {kept_output}; rule tổng hợp -> {rules_output}")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_role_client(service_account_json, env_file, model_arg, role: str, input_path,
                         location: str = "local", base_url_override: Optional[str] = None):
    """Ủy quyền cho auto_model_relay.resolve_role_client() (nguồn DUY NHẤT xử lý credential)."""
    return auto_model_relay.resolve_role_client(
        role,
        service_account_json=service_account_json,
        env_file=env_file,
        location=location,
        model=model_arg,
        base_url=base_url_override,
        input_path=input_path,
    )


# =============================================================================================
# CLI
# =============================================================================================

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("filter-qa", help="Lọc chất lượng câu hỏi của 1 task bằng 1 provider (model mạnh).")
    env_paths.add_location_arg(p)
    p.add_argument("--task", required=True, choices=sorted(_TASK_SPECIFIC))
    p.add_argument("--pre-final", required=True, help="File pre-final JSONL (input).")
    p.add_argument("--kept-output", required=True, help="File JSONL các câu được GIỮ (output).")
    p.add_argument("--rules-output", required=True, help="File JSON các criteria_summary mỗi batch.")
    p.add_argument("--provider", required=True, choices=["gemini", "openai"])
    p.add_argument("--model", default=None, help="Override model (mặc định: model mạnh theo provider).")
    p.add_argument("--service-account-json", default=None, help="Vertex AI (provider gemini).")
    p.add_argument("--env-file", default=None, help='File .env local dạng KEY="value" # base_url # model.')
    p.add_argument("--gemini-base-url", default=None, help="Override base_url Gemini (khi dùng proxy OpenAI-compatible).")
    p.add_argument("--openai-api-key-file", default=None, help="File key OpenAI-compatible (provider openai).")
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    args = parser.parse_args(argv)

    if args.command == "filter-qa":
        input_path = Path(args.pre_final)
        env_file = args.env_file or (str(env_paths.default_env_file(args.location)) if args.location == "local" else None)

        if args.provider == "gemini":
            service_account_json = args.service_account_json or (
                str(env_paths.gemini_service_account_path(args.location)) if args.location == "drive" and not env_file else None
            )
            client, model = _resolve_role_client(service_account_json, env_file, args.model, "GEMINI", input_path,
                                                 location=args.location, base_url_override=args.gemini_base_url)
        else:
            if args.openai_api_key_file:
                client = auto_model_relay.load_openai_client(
                    Path(args.openai_api_key_file), args.openai_base_url or auto_model_relay.DEFAULT_OPENAI_BASE_URL
                )
                model = args.model or DEFAULT_OPENAI_FILTER_MODEL
            else:
                client, model = _resolve_role_client(None, env_file, args.model, "OPENAI", input_path, location=args.location)

        provider_call = make_provider_call(args.provider, client, model)
        system_prompt = build_filter_system_prompt(args.task)
        context_fields = CONTEXT_FIELDS[args.task]
        records = _load_jsonl(input_path)
        print(f"[{args.task}/{args.provider}:{model}] Đọc {len(records)} câu từ {input_path}")

        filter_questions(
            provider_call, records, Path(args.kept_output), Path(args.rules_output),
            system_prompt=system_prompt, context_fields=context_fields,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
