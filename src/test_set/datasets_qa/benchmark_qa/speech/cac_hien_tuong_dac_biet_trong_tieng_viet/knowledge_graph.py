"""Shared schema/loader for the versioned "knowledge graph" JSON.

The knowledge graph is produced ONCE (or rarely, when refreshed) by manual_model_relay.py's
manual copy-paste relay between 3 free web-tier chat models (Gemini/ChatGPT/Claude), and consumed
cheaply/repeatedly by classify-levels/generate-questions in han_viet_pipeline.py,
tu_muon_pipeline.py, tu_lay_pipeline.py, phuong_ngu_pipeline.py. Kept as its own tiny module
(not duplicated 4x) because this schema is the contract every pipeline's --knowledge-json flag
relies on -- a change here must stay in sync with every consumer.

Schema (see also apply_knowledge_graph.py, which writes "words" back onto a candidate-word CSV):
{
  "schema_version": "1.0",
  "task": "tu_lay", "variant": "toan_bo" | null,
  "generated_at": "<ISO8601>",
  "relay_meta": {"model_order": [...], "rounds_completed": int, "max_rounds": int,
                 "stopped_reason": "consensus" | "max_rounds", "consensus_round": int | null},
  "words": [
    {"word": str, "status": "confirmed" | "corrected" | "added" | "rejected",
     "source": "existing_csv" | "web_research" | "corpus_scan" | "model_proposed",
     "fields": {...task-specific...}, "extra_fields": {...}, "notes": str}
  ],
  "rules": {"level_1": {...}, "level_2": {...}, "level_3": {...}, "cloze_toan_bo": {...},
            "van_new_mechanic": {"enabled": bool, ...}, "chung_new_mechanic": {"enabled": bool, ...}}
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0"


def load_knowledge_graph(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        kg = json.load(f)
    if kg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version={kg.get('schema_version')!r}, script này chỉ hiểu "
            f"{SCHEMA_VERSION!r} -- knowledge graph có thể đã được tạo bởi 1 bản relay tool cũ/mới hơn."
        )
    return kg


def words_index(kg: dict) -> dict[str, dict]:
    """word -> {status, source, fields, extra_fields, notes}. Từ có status=="rejected" bị loại
    hoàn toàn khỏi index -- 1 từ bị debate bác bỏ không được để lọt lại downstream dù vẫn còn
    nằm trong "words" (giữ nguyên record đó chỉ để có audit trail, không phải để dùng)."""
    return {w["word"]: w for w in kg.get("words", []) if w.get("status") != "rejected"}


def rule(kg: dict, key: str) -> dict:
    return kg.get("rules", {}).get(key, {})


def rule_addendum_text(kg: Optional[dict], key: str) -> str:
    """Trả về đoạn text bổ sung vào system prompt hiện có, lấy từ rules[key]['description'].
    kg=None hoặc key vắng -> "" -- mọi nơi gọi hàm này đều viết dạng
    `SYSTEM_PROMPT + rule_addendum_text(kg, "level_3")`, nên khi không truyền --knowledge-json,
    hành vi giữ NGUYÊN y hệt trước khi có knowledge graph (an toàn ngược)."""
    if kg is None:
        return ""
    description = rule(kg, key).get("description")
    if not description:
        return ""
    return "\n\nBỔ SUNG (từ đồ thị tri thức đã được 3 model rà soát):\n" + description
