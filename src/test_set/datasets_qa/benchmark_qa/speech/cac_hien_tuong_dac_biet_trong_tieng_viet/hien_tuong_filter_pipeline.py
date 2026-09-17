"""Build "Hiện tượng đặc biệt trong tiếng Việt" filtered word lists + matched ASR samples.

Extracted from noteboooks/Hien_tuong_dac_biet_trong_tieng_Viet.ipynb -- this is the LOCAL
preprocessing stage shared by all 4 tasks (Từ mượn, Từ láy toàn bộ/vần/chung): it takes a
candidate word list CSV (curated by hand -- e.g. loanwords or reduplicative words) and the raw
ASR transcript sources, and produces (a) a "final" CSV containing only the words that actually
appear somewhere in the ASR transcripts, and (b) a JSON array of ASR samples annotated with
which candidate word(s) were found in each sample's transcript, with full CSV metadata attached.

Two transcript sources, combinable (UNION, not either/or):
  - vlsp/*.json + asr_source_ledger.json: LOCAL ONLY (stuff/ is gitignored, never pushed to
    GitHub) -- Colab has no access to them. Run --vlsp-glob/--ledger here, then upload the
    resulting --out-csv/--out-json files to Google Drive for Colab to consume.
  - full_transcripts.json (--full-transcripts-json): a Drive-resident copy of
    stuff/benchmark_qa/release_hf_transcripts_by_dataset.json (20 dataset, ~35k transcript,
    broader dataset coverage than vlsp+ledger's ~12 dataset) -- because this file lives on
    Drive, not in gitignored stuff/, every subcommand below can now ALSO run directly on Colab
    when only --full-transcripts-json is given (no local-only step required).

6 subcommands:
  1. filter-csv            -- keep only candidate-CSV rows whose word/phrase appears (word-
                               boundary match) somewhere in the ASR transcript corpus.
  2. build-samples          -- for the "final" (filtered) CSV, scan every ASR sample's transcript
                               and attach full CSV metadata for every candidate word found in it.
  3. split-by-prefix        -- split a "final" CSV (and its build-samples output, if given) into
                               2 groups by whether a given column's value STARTS WITH a given
                               prefix (used to split từ láy into "toàn bộ" vs "vần" by the
                               "Phân loại" column).
  4. dedupe-samples         -- remove words from a samples file's per-sample word list if that
                               SAME word already appears in one or more OTHER samples files (used
                               to keep the "chung" từ láy samples from re-covering words already
                               handled by the "toàn bộ"/"vần" samples); drops samples left with an
                               empty list.
  5. word-coverage-report   -- per-word hit count/dataset breakdown/example transcripts across
                               the scanned corpus -- evidence for build_debate_seed.py, and a way
                               to see which candidate words are dead weight (0 hits) before
                               spending a debate round on them.
  6. build-cloze-samples    -- (từ láy toàn bộ only) keep samples containing a word's BASE
                               (un-reduplicated) syllable but NOT its full reduplicated form --
                               prerequisite for tu_lay_pipeline.py's generate-cloze-questions.

Usage (Từ mượn):
    python hien_tuong_filter_pipeline.py filter-csv \\
        --src-csv tu_muon_tieng_viet_viet_hoa.csv --word-col "Từ Tiếng Việt (Việt Hóa)" \\
        --vlsp-glob "asr_datasets/vlsp/*.json" --ledger asr_datasets/asr_source_ledger.json \\
        --out-csv tu_muon_tieng_viet_viet_hoa_final.csv

    python hien_tuong_filter_pipeline.py build-samples \\
        --csv tu_muon_tieng_viet_viet_hoa_final.csv --word-col "Từ Tiếng Việt (Việt Hóa)" \\
        --vlsp-glob "asr_datasets/vlsp/*.json" --ledger asr_datasets/asr_source_ledger.json \\
        --list-key tu_muon_xuat_hien \\
        --metadata-cols '{"tu": "Từ Tiếng Việt (Việt Hóa)", "tu_goc": "Từ Gốc", "ngon_ngu_nguon_goc": "Ngôn Ngữ / Nguồn Gốc", "nhom_linh_vuc": "Nhóm Lĩnh Vực", "y_nghia_ghi_chu": "Ý Nghĩa / Ghi Chú"}' \\
        --out-json asr_datasets/asr_samples_with_tu_muon.json

Usage (Từ láy toàn bộ + vần, split từ 1 CSV/samples chung):
    python hien_tuong_filter_pipeline.py filter-csv \\
        --src-csv tu_lay_toan_bo_va_van.csv --word-col "Từ láy" \\
        --vlsp-glob "asr_datasets/vlsp/*.json" --ledger asr_datasets/asr_source_ledger.json \\
        --out-csv tu_lay_toan_bo_va_van_final.csv

    python hien_tuong_filter_pipeline.py build-samples \\
        --csv tu_lay_toan_bo_va_van_final.csv --word-col "Từ láy" \\
        --vlsp-glob "asr_datasets/vlsp/*.json" --ledger asr_datasets/asr_source_ledger.json \\
        --list-key tu_lay_xuat_hien \\
        --metadata-cols '{"tu": "Từ láy", "phan_loai": "Phân loại", "y_nghia": "Ý nghĩa", "sac_thai_bieu_dat": "Sắc thái biểu đạt"}' \\
        --out-json asr_datasets/asr_samples_with_tu_lay_toan_bo_va_van.json

    python hien_tuong_filter_pipeline.py split-by-prefix \\
        --csv tu_lay_toan_bo_va_van_final.csv --column "Phân loại" --prefix "Láy toàn bộ" \\
        --samples asr_datasets/asr_samples_with_tu_lay_toan_bo_va_van.json --word-col "Từ láy" --list-key tu_lay_xuat_hien --metadata-word-key tu \\
        --out-csv-matched tu_lay_toan_bo_final.csv --out-csv-rest tu_lay_van_final.csv \\
        --out-samples-matched asr_datasets/asr_samples_with_tu_lay_toan_bo.json --out-samples-rest asr_datasets/asr_samples_with_tu_lay_van.json

Usage (Từ láy "chung" -- bộ nguồn RIÊNG, dedupe với 2 bộ trên):
    python hien_tuong_filter_pipeline.py filter-csv --src-csv tu_lay_tieng_viet.csv --word-col "Từ láy" ... --out-csv tu_lay_tieng_viet_final.csv
    python hien_tuong_filter_pipeline.py build-samples --csv tu_lay_tieng_viet_final.csv --word-col "Từ láy" --list-key tu_lay_xuat_hien \\
        --metadata-cols '{"tu": "Từ láy", "loai_tu_lay": "Loại từ láy", "tu_loai": "Từ loại", "y_nghia": "Ý nghĩa", "sac_thai_bieu_dat": "Sắc thái biểu đạt"}' \\
        --out-json asr_datasets/asr_samples_with_tu_lay.json
    python hien_tuong_filter_pipeline.py dedupe-samples \\
        --samples asr_datasets/asr_samples_with_tu_lay.json --list-key tu_lay_xuat_hien --word-key tu \\
        --against asr_datasets/asr_samples_with_tu_lay_toan_bo.json --against asr_datasets/asr_samples_with_tu_lay_van.json \\
        --output asr_datasets/asr_samples_with_tu_lay.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.lower()


def word_appears_in_corpus(word: str, corpus: str) -> bool:
    pattern = r"(?<![\wÀ-ỹ])" + re.escape(normalize_for_match(word)) + r"(?![\wÀ-ỹ])"
    return re.search(pattern, corpus) is not None


def _load_transcripts_from_vlsp_jsonl(path: Path) -> list[str]:
    # vlsp*.json là JSONL (mỗi dòng 1 JSON object, field "transcript"), KHÔNG phải 1 JSON array.
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("transcript"):
                texts.append(obj["transcript"])
    return texts


def _load_transcripts_from_ledger(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["transcript"] for item in data if item.get("transcript")]


def _load_samples_from_vlsp_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("transcript"):
                obj["source_file"] = path.name
                samples.append(obj)
    return samples


def _load_samples_from_ledger(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        if item.get("transcript"):
            item = dict(item)
            item["source_file"] = path.name
            samples.append(item)
    return samples


def _load_samples_from_full_transcripts(path: Path) -> list[dict]:
    """full_transcripts.json chấp nhận CẢ 2 dạng:
      (a) list phẳng [{"dataset": str, "transcript": str, ...}, ...] -- dạng THẬT của file này
          trên Drive/local hiện tại (94k+ entry, "dataset" là field trong từng object).
      (b) dict {dataset_name: [{"audio": str, "transcript": str}, ...]} -- dạng
          release_hf_transcripts_by_dataset.json gốc, giữ tương thích ngược nếu ai đó dùng bản
          đó thay vì bản list phẳng."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    if isinstance(data, list):
        for item in data:
            if item.get("transcript"):
                samples.append({
                    "transcript": item["transcript"], "audio_filepath": item.get("audio"),
                    "dataset": item.get("dataset", "unknown"), "source_file": item.get("source_file", path.name),
                })
        return samples

    for dataset_name, entries in data.items():
        for item in entries:
            if item.get("transcript"):
                samples.append({
                    "transcript": item["transcript"], "audio_filepath": item.get("audio"),
                    "dataset": dataset_name, "source_file": path.name,
                })
    return samples


def load_corpus_and_samples(
    vlsp_glob: Optional[str] = None, ledger_path: Optional[Path] = None,
    full_transcripts_json: Optional[Path] = None,
) -> tuple[str, list[dict]]:
    """Trả về (corpus_normalized, all_samples) -- dùng chung cho filter-csv/build-samples/
    word-coverage-report/build-cloze-samples để đảm bảo mọi bước luôn thấy CÙNG 1 nguồn
    transcript. Cần ít nhất 1 trong 2 nguồn: (vlsp_glob + ledger_path) và/hoặc
    full_transcripts_json -- gọi CẢ HAI để quét UNION của 2 phạm vi (mở rộng, không thay thế
    phạm vi vlsp+ledger cũ)."""
    if not (vlsp_glob and ledger_path) and not full_transcripts_json:
        raise ValueError("Cần ít nhất 1 nguồn: (--vlsp-glob + --ledger) và/hoặc --full-transcripts-json.")

    all_samples: list[dict] = []
    n_vlsp_files = 0
    if vlsp_glob and ledger_path:
        vlsp_paths = sorted(glob.glob(vlsp_glob))
        n_vlsp_files = len(vlsp_paths)
        for p in vlsp_paths:
            all_samples.extend(_load_samples_from_vlsp_jsonl(Path(p)))
        all_samples.extend(_load_samples_from_ledger(ledger_path))
        print(f"Đã load {len(all_samples)} sample từ {n_vlsp_files} file vlsp + 1 file ledger.")

    if full_transcripts_json:
        n_before = len(all_samples)
        all_samples.extend(_load_samples_from_full_transcripts(full_transcripts_json))
        print(f"Đã load thêm {len(all_samples) - n_before} sample từ {full_transcripts_json.name}.")

    corpus_normalized = "\n".join(normalize_for_match(s["transcript"]) for s in all_samples)
    print(f"Tổng: {len(all_samples)} sample trong corpus quét.")
    return corpus_normalized, all_samples


# ============================================================================================
# Subcommand 1: filter-csv
# ============================================================================================

def filter_csv(src_csv: Path, word_col: str, corpus_normalized: str, out_csv: Path) -> None:
    with open(src_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    kept_rows = [r for r in rows if r[word_col].strip() and word_appears_in_corpus(r[word_col].strip(), corpus_normalized)]
    print(f"Giữ lại {len(kept_rows)}/{len(rows)} dòng có xuất hiện trong transcript ASR.")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    print(f"Đã lưu {out_csv} ({len(kept_rows)} dòng).")


# ============================================================================================
# Subcommand 2: build-samples
# ============================================================================================

def build_samples(
    csv_path: Path, word_col: str, all_samples: list[dict], corpus_normalized: str,
    metadata_cols: dict[str, str], list_key: str, out_json: Path,
) -> None:
    word_rows: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[word_col].strip()
            if w:
                word_rows[w] = row

    # Sắp xếp dài -> ngắn: nếu 1 câu chứa cả từ dài và từ ngắn là con của nó, ưu tiên báo cáo từ
    # dài/cụ thể hơn trước (không đổi kết quả matching, chỉ đổi thứ tự trong list).
    words_sorted = sorted(word_rows, key=len, reverse=True)
    print(f"Đã load {len(words_sorted)} từ đã lọc từ {csv_path}.")

    matched_samples = []
    for sample in all_samples:
        transcript_norm = normalize_for_match(sample["transcript"])
        found_words = [w for w in words_sorted if word_appears_in_corpus(w, transcript_norm)]
        if not found_words:
            continue
        new_sample = dict(sample)
        new_sample[list_key] = [
            {json_key: word_rows[w][csv_col] for json_key, csv_col in metadata_cols.items()}
            for w in found_words
        ]
        matched_samples.append(new_sample)

    print(f"Giữ lại {len(matched_samples)}/{len(all_samples)} sample có chứa ít nhất 1 từ đã lọc.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(matched_samples, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {out_json} ({len(matched_samples)} sample).")


# ============================================================================================
# Subcommand 3: split-by-prefix
# ============================================================================================

def split_by_prefix(
    csv_path: Path, column: str, prefix: str, out_csv_matched: Path, out_csv_rest: Path,
    samples_path: Optional[Path], word_col: str, list_key: Optional[str], metadata_word_key: Optional[str],
    out_samples_matched: Optional[Path], out_samples_rest: Optional[Path],
) -> None:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    matched_rows = [r for r in rows if r[column].strip().startswith(prefix)]
    rest_rows = [r for r in rows if not r[column].strip().startswith(prefix)]
    print(f"Tách CSV theo cột {column!r} bắt đầu bằng {prefix!r}: {len(matched_rows)} khớp, "
          f"{len(rest_rows)} còn lại (/{len(rows)} tổng).")

    for path, out_rows in ((out_csv_matched, matched_rows), (out_csv_rest, rest_rows)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Đã lưu {path} ({len(out_rows)} dòng).")

    if samples_path is None:
        return

    matched_words = {r[word_col].strip() for r in matched_rows}
    rest_words = {r[word_col].strip() for r in rest_rows}

    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)

    def _filter_samples(word_set: set[str]) -> list[dict]:
        out = []
        for s in samples:
            found = [item for item in s.get(list_key, []) if item[metadata_word_key] in word_set]
            if found:
                new_s = dict(s)
                new_s[list_key] = found
                out.append(new_s)
        return out

    matched_samples = _filter_samples(matched_words)
    rest_samples = _filter_samples(rest_words)
    print(f"Tách sample tương ứng: {len(matched_samples)} khớp, {len(rest_samples)} còn lại "
          f"(/{len(samples)} tổng).")

    with open(out_samples_matched, "w", encoding="utf-8") as f:
        json.dump(matched_samples, f, ensure_ascii=False, indent=2)
    with open(out_samples_rest, "w", encoding="utf-8") as f:
        json.dump(rest_samples, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {out_samples_matched} ({len(matched_samples)} sample), "
          f"{out_samples_rest} ({len(rest_samples)} sample).")


# ============================================================================================
# Subcommand 4: dedupe-samples
# ============================================================================================

def dedupe_samples(samples_path: Path, list_key: str, word_key: str, against_paths: list[Path], output_path: Path) -> None:
    covered_words: set[str] = set()
    for p in against_paths:
        with open(p, encoding="utf-8") as f:
            other_samples = json.load(f)
        for s in other_samples:
            for item in s.get(list_key, []):
                covered_words.add(item[word_key])
    print(f"Tổng số từ đã cover (từ {len(against_paths)} file khác): {len(covered_words)}")

    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Số sample gốc: {len(samples)}")

    kept = []
    for s in samples:
        remaining = [item for item in s.get(list_key, []) if item[word_key] not in covered_words]
        if remaining:
            new_s = dict(s)
            new_s[list_key] = remaining
            kept.append(new_s)

    print(f"Số sample còn lại sau khi loại: {len(kept)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {output_path}.")


# ============================================================================================
# Subcommand 5: word-coverage-report
# ============================================================================================

def word_coverage_report(
    words: list[str], all_samples: list[dict], corpus_normalized: str, *, max_examples_per_word: int = 3,
) -> dict[str, dict]:
    """Với mỗi từ candidate, đếm số lần xuất hiện + phân bố theo dataset + vài transcript ví dụ.
    Dùng để (a) cấp bằng chứng THẬT cho build_debate_seed.py (không bịa), (b) lộ ra ngay từ nào
    0 hit trong TOÀN BỘ corpus đã quét (kể cả sau khi mở rộng full_transcripts.json) -- những từ
    này không đáng đưa vào 1 vòng debate, và (c) định lượng full_transcripts.json thực sự tăng
    thêm bao nhiêu sample dùng được cho mỗi từ (kiểm tra sizing benchmark)."""
    report: dict[str, dict] = {}
    for w in words:
        w_norm = normalize_for_match(w)
        hits_by_dataset: dict[str, int] = {}
        examples: list[str] = []
        for s in all_samples:
            transcript_norm = normalize_for_match(s["transcript"])
            if not word_appears_in_corpus(w, transcript_norm):
                continue
            dataset = s.get("dataset") or s.get("source_file") or "unknown"
            hits_by_dataset[dataset] = hits_by_dataset.get(dataset, 0) + 1
            if len(examples) < max_examples_per_word:
                examples.append(s["transcript"])
        report[w] = {
            "total_hits": sum(hits_by_dataset.values()),
            "hits_by_dataset": hits_by_dataset,
            "example_transcripts": examples,
        }

    n_zero_hit = sum(1 for v in report.values() if v["total_hits"] == 0)
    print(f"Đã kiểm tra {len(words)} từ trong {len(all_samples)} sample -- "
          f"{n_zero_hit} từ KHÔNG xuất hiện dù chỉ 1 lần.")
    return report


# ============================================================================================
# Subcommand 6: build-cloze-samples (chỉ dùng cho từ láy toàn bộ -- xem tu_lay_pipeline.py)
# ============================================================================================

def build_cloze_samples(
    csv_path: Path, word_col: str, base_word_col: str, all_samples: list[dict], corpus_normalized: str,
    list_key: str, out_json: Path,
) -> None:
    """Với mỗi dòng CSV từ láy toàn bộ, giữ sample có TỪ GỐC (chưa láy, base_word_col) trong
    transcript NHƯNG KHÔNG có từ láy đầy đủ (word_col) -- nếu transcript đã chứa sẵn từ láy đầy
    đủ thì câu hỏi cloze (điền vào chỗ trống) sẽ lộ đáp án ngay trong audio, làm câu hỏi vô
    nghĩa. Dùng CHÍNH word_appears_in_corpus đã có, không viết matcher riêng."""
    word_rows: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[word_col].strip()
            base = row.get(base_word_col, "").strip()
            if w and base:
                word_rows[w] = row

    print(f"Đã load {len(word_rows)} từ láy toàn bộ có cột {base_word_col!r} từ {csv_path}.")

    matched_samples = []
    for sample in all_samples:
        transcript_norm = normalize_for_match(sample["transcript"])
        found = [
            w for w, row in word_rows.items()
            if word_appears_in_corpus(row[base_word_col].strip(), transcript_norm)
            and not word_appears_in_corpus(w, transcript_norm)
        ]
        if not found:
            continue
        new_sample = dict(sample)
        new_sample[list_key] = [
            {"tu": w, "base_word": word_rows[w][base_word_col], "phan_loai": word_rows[w].get("Phân loại", ""),
             "y_nghia": word_rows[w].get("Ý nghĩa", ""), "sac_thai_bieu_dat": word_rows[w].get("Sắc thái biểu đạt", "")}
            for w in found
        ]
        matched_samples.append(new_sample)

    print(f"Giữ lại {len(matched_samples)}/{len(all_samples)} sample có từ GỐC (chưa láy) "
          f"nhưng KHÔNG có từ láy đầy đủ trong transcript.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(matched_samples, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {out_json} ({len(matched_samples)} sample).")


# ============================================================================================
# CLI
# ============================================================================================

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("filter-csv")
    p1.add_argument("--src-csv", required=True)
    p1.add_argument("--word-col", required=True)
    p1.add_argument("--vlsp-glob", default=None)
    p1.add_argument("--ledger", default=None)
    p1.add_argument("--full-transcripts-json", default=None, help="Bản Drive của release_hf_transcripts_by_dataset.json -- quét THÊM (không thay) vlsp+ledger.")
    p1.add_argument("--out-csv", required=True)

    p2 = sub.add_parser("build-samples")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--word-col", required=True)
    p2.add_argument("--vlsp-glob", default=None)
    p2.add_argument("--ledger", default=None)
    p2.add_argument("--full-transcripts-json", default=None)
    p2.add_argument("--metadata-cols", required=True, help='JSON object {"json_key": "csv_column_name", ...}')
    p2.add_argument("--list-key", required=True, help='vd "tu_muon_xuat_hien" hoặc "tu_lay_xuat_hien"')
    p2.add_argument("--out-json", required=True)

    p3 = sub.add_parser("split-by-prefix")
    p3.add_argument("--csv", required=True)
    p3.add_argument("--column", required=True)
    p3.add_argument("--prefix", required=True)
    p3.add_argument("--out-csv-matched", required=True)
    p3.add_argument("--out-csv-rest", required=True)
    p3.add_argument("--samples", default=None)
    p3.add_argument("--word-col", default=None)
    p3.add_argument("--list-key", default=None)
    p3.add_argument("--metadata-word-key", default=None)
    p3.add_argument("--out-samples-matched", default=None)
    p3.add_argument("--out-samples-rest", default=None)

    p4 = sub.add_parser("dedupe-samples")
    p4.add_argument("--samples", required=True)
    p4.add_argument("--list-key", required=True)
    p4.add_argument("--word-key", required=True)
    p4.add_argument("--against", action="append", required=True, help="Lặp lại cờ này cho mỗi file cần loại trừ.")
    p4.add_argument("--output", required=True)

    p5 = sub.add_parser("word-coverage-report", help="Đếm số lần mỗi từ candidate xuất hiện trong corpus -- bằng chứng cho debate seed.")
    p5.add_argument("--csv", default=None)
    p5.add_argument("--word-col", default=None)
    p5.add_argument("--words-json", default=None, help='Thay thế --csv/--word-col: file JSON list[str] các từ (dùng khi chưa có CSV, ví dụ Hán Việt).')
    p5.add_argument("--vlsp-glob", default=None)
    p5.add_argument("--ledger", default=None)
    p5.add_argument("--full-transcripts-json", default=None)
    p5.add_argument("--max-examples-per-word", type=int, default=3)
    p5.add_argument("--out-json", required=True)

    p6 = sub.add_parser("build-cloze-samples", help="Chỉ dùng cho từ láy toàn bộ -- xem tu_lay_pipeline.py generate-cloze-questions.")
    p6.add_argument("--csv", required=True)
    p6.add_argument("--word-col", required=True)
    p6.add_argument("--base-word-col", required=True)
    p6.add_argument("--vlsp-glob", default=None)
    p6.add_argument("--ledger", default=None)
    p6.add_argument("--full-transcripts-json", default=None)
    p6.add_argument("--list-key", required=True)
    p6.add_argument("--out-json", required=True)

    args = parser.parse_args(argv)

    if args.command == "filter-csv":
        corpus, _ = load_corpus_and_samples(args.vlsp_glob, Path(args.ledger) if args.ledger else None, Path(args.full_transcripts_json) if args.full_transcripts_json else None)
        filter_csv(Path(args.src_csv), args.word_col, corpus, Path(args.out_csv))

    elif args.command == "build-samples":
        corpus, all_samples = load_corpus_and_samples(args.vlsp_glob, Path(args.ledger) if args.ledger else None, Path(args.full_transcripts_json) if args.full_transcripts_json else None)
        metadata_cols = json.loads(args.metadata_cols)
        build_samples(Path(args.csv), args.word_col, all_samples, corpus, metadata_cols, args.list_key, Path(args.out_json))

    elif args.command == "split-by-prefix":
        split_by_prefix(
            Path(args.csv), args.column, args.prefix, Path(args.out_csv_matched), Path(args.out_csv_rest),
            Path(args.samples) if args.samples else None, args.word_col, args.list_key, args.metadata_word_key,
            Path(args.out_samples_matched) if args.out_samples_matched else None,
            Path(args.out_samples_rest) if args.out_samples_rest else None,
        )

    elif args.command == "dedupe-samples":
        dedupe_samples(Path(args.samples), args.list_key, args.word_key, [Path(p) for p in args.against], Path(args.output))

    elif args.command == "word-coverage-report":
        corpus, all_samples = load_corpus_and_samples(args.vlsp_glob, Path(args.ledger) if args.ledger else None, Path(args.full_transcripts_json) if args.full_transcripts_json else None)
        if args.words_json:
            with open(args.words_json, encoding="utf-8") as f:
                words = json.load(f)
        else:
            assert args.csv and args.word_col, "Cần --csv + --word-col, hoặc --words-json."
            with open(args.csv, encoding="utf-8-sig", newline="") as f:
                words = [r[args.word_col].strip() for r in csv.DictReader(f) if r[args.word_col].strip()]
        report = word_coverage_report(words, all_samples, corpus, max_examples_per_word=args.max_examples_per_word)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu {args.out_json}.")

    elif args.command == "build-cloze-samples":
        corpus, all_samples = load_corpus_and_samples(args.vlsp_glob, Path(args.ledger) if args.ledger else None, Path(args.full_transcripts_json) if args.full_transcripts_json else None)
        build_cloze_samples(Path(args.csv), args.word_col, args.base_word_col, all_samples, corpus, args.list_key, Path(args.out_json))


if __name__ == "__main__":
    main()
