"""Assembles TRANSLATION_QUALITY_REPORT.md from live translation results.

Called from noteboooks/Translate_QA_Datasets.ipynb after the notebook has
computed mb_quality/mq_quality (mục 11), mb_merged/mq_merged (mục 9),
mb_cost_time/mq_cost_time + mb_extrap_10m/mq_extrap_10m (mục 12),
mb_failed/mq_failed (mục 14) -- see build_report_markdown()'s parameters for
the exact shape each one needs.

Editing the report's FORMAT (wording, section order, what gets shown) means
editing this file, not the notebook -- the notebook cell just gathers the
already-computed variables and calls save_report().
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def render_stats_table(mb_quality: dict, mq_quality: dict) -> str:
    rows = [
        "| Dataset | Tổng sample đã thử dịch | Dịch OK | Dịch lỗi (đã loại) | Tỉ lệ thành công |",
        "|---|---|---|---|---|",
    ]
    for name, q in [("MusicBench", mb_quality), ("MusicQA", mq_quality)]:
        rows.append(f"| {name} | {q['n_total']} | {q['n_ok']} | {q['n_failed']} | {q['success_rate']*100:.1f}% |")
    return "\n".join(rows)


def render_cost_time_table(mb_cost_time: Optional[dict], mq_cost_time: Optional[dict]) -> str:
    rows = [
        "| Dataset | Sample dịch (lần chạy này) | Thời gian ước tính /1000 sample | Chi phí ước tính /1000 sample |",
        "|---|---|---|---|",
    ]
    for name, ct in [("MusicBench", mb_cost_time), ("MusicQA", mq_cost_time)]:
        if ct is None:
            rows.append(f"| {name} | 0 (đã resume-skip hết) | - | - |")
        else:
            rows.append(f"| {name} | {ct['n']} | {ct['time_per_1k_min']:.1f} phút | ${ct['cost_per_1k']:.4f} |")
    return "\n".join(rows)


def render_extrap_table(mb_extrap_10m: Optional[dict], mq_extrap_10m: Optional[dict]) -> str:
    rows = [
        "| Dataset | Chi phí ước tính /10 triệu sample | Thời gian ước tính /10 triệu sample |",
        "|---|---|---|",
    ]
    for name, e in [("MusicBench", mb_extrap_10m), ("MusicQA", mq_extrap_10m)]:
        if e is None:
            rows.append(f"| {name} | | |")
        else:
            rows.append(f"| {name} | ${e['cost_usd']:,.2f} | {e['time_hours']:,.1f} giờ |")
    return "\n".join(rows)


def render_sample_blocks(mb_merged: list[dict], mq_merged: list[dict], n: int = 3) -> str:
    lines = ["**MusicBench**", ""]
    shown = 0
    for rec in mb_merged:
        if "main_caption_vi" not in rec:
            continue
        lines.append(f"> EN: {rec['main_caption']}")
        lines.append(">")
        lines.append(f"> VI: {rec['main_caption_vi']}")
        lines.append("")
        shown += 1
        if shown >= n:
            break

    lines.append("")
    lines.append("**MusicQA**")
    lines.append("")
    shown = 0
    for rec in mq_merged:
        if "question_vi" not in rec:
            continue
        lines.append(f"> Q (EN): {rec['conversation'][0]['value']}")
        lines.append(f"> Q (VI): {rec['question_vi']}")
        lines.append(">")
        lines.append(f"> A (EN): {rec['conversation'][1]['value']}")
        lines.append(f"> A (VI): {rec['answer_vi']}")
        lines.append("")
        shown += 1
        if shown >= n:
            break
    return "\n".join(lines).rstrip()


def render_failed_section(name: str, failed: list[dict]) -> str:
    lines = [f"### {name} -- sample dịch lỗi ({len(failed)})", ""]
    if not failed:
        lines.append("_Không có sample lỗi nào._")
        return "\n".join(lines)
    for item in failed:
        lines.append(f"- **sample_index={item['sample_index']}** -- lỗi: {item['error']}")
        for field_name, text in item["source_fields"].items():
            lines.append(f"  - `{field_name}` (EN gốc): {text}")
            attempted = (item.get("attempted_fields") or {}).get(field_name)
            if attempted is not None:
                lines.append(f"  - `{field_name}` (bản dịch bị từ chối): {attempted}")
    return "\n".join(lines)


REPORT_HEADER_AND_SECTION1 = """# Báo cáo chất lượng dịch EN → VI (MusicBench, MusicQA)

Báo cáo mô tả cách đánh giá chất lượng bản dịch tự động (Gemini/Vertex AI, qua
`translate_dataset.py`) cho 2 dataset `MusicBench` (caption nhạc) và `MusicQA`
(hỏi-đáp về nhạc), và thống kê kết quả trên dữ liệu đã dịch.

## 1. Tiêu chí đánh giá

Chất lượng được kiểm tra ở 2 lớp: **lúc dịch** (tự động loại sample lỗi ngay
khi phát hiện, không đợi hậu kiểm) và **hậu kiểm** (thống kê lại toàn bộ kết
quả sau khi chạy xong để phát hiện xu hướng lỗi hệ thống nếu có).

### 1.1 Kiểm tra lúc dịch (`translate_dataset.py`)

| Tiêu chí | Cách kiểm tra | Xử lý khi fail |
|---|---|---|
| Vẫn là tiếng Anh | Ưu tiên đếm **tỉ lệ từ có dấu tiếng Việt** trước (câu ≤6 từ chỉ cần 1 từ có dấu; câu dài hơn cần ≥15% số từ có dấu) — tín hiệu này đáng tin hơn `langdetect` cho câu pha trộn nhiều thuật ngữ/tên riêng tiếng Anh có chủ đích (tên nhạc cụ, thể loại). Chỉ dùng `langdetect.detect() == "vi"` làm phương án dự phòng khi câu hoàn toàn không có dấu. **Ngoại lệ**: `_looks_like_term_list()` bỏ qua hẳn bước này nếu câu trả lời là tên thể loại/nhạc cụ ngắn hoặc danh sách phân tách bằng dấu phẩy/gạch chéo, mỗi phần ≤3 từ (vd `"Guitar."`, `"Electronic, hard rock, metal"`) — tiếng Việt mượn nguyên các từ này nên giữ tiếng Anh là bản dịch **đúng**, không phải bỏ sót | Dịch lại riêng sample đó, tối đa `MAX_RETRIES` lần |
| Tỉ lệ độ dài bất thường | `len(bản dịch) / len(bản gốc)` phải nằm trong `[0.3, 2.0]` — bắt các case rỗng/bị cắt/dịch lan man thêm nội dung. **Ngoại lệ cho câu gốc ngắn** (≤40 ký tự): chỉ nới lỏng giới hạn **trên** thành "dài hơn bản gốc tối đa 40 ký tự" (số tuyệt đối, không phải tỉ lệ) — vì câu hỏi Yes/No tiếng Việt (`"...có... không?"`) tự nhiên dài hơn hẳn bản tiếng Anh gốc dù dịch đúng; giới hạn **dưới** (chặn cắt cụt) không đổi trong mọi trường hợp | Dịch lại riêng sample đó |
| Response JSON hỏng/thiếu id | Không `json.loads` được, hoặc id/field trả về không khớp với batch gửi đi | Tách batch, dịch lại từng sample riêng lẻ thay vì hỏng cả batch |
| Câu hỏi và câu trả lời lệch nghĩa nhau | Không có bộ kiểm tra tự động (khó verify bằng rule đơn giản) — phòng ngừa bằng thiết kế: `question` + `answer` của cùng 1 sample luôn được gộp thành **1 "unit"**, dịch chung trong **cùng 1 lượt gọi Gemini** để model thấy cả hai và giữ ngữ cảnh nhất quán | N/A — đây là biện pháp phòng ngừa ở bước dịch, không phải bộ lọc hậu kiểm |

Sample dịch lỗi sau khi đã thử lại đủ số lần bị **loại hẳn** khỏi file kết quả
cuối (`*_translated.json`) — không có sample nào "dịch sai vẫn giữ lại".

### 1.2 Hậu kiểm (thống kê trên toàn bộ kết quả)

Sau khi dịch xong, tính lại trên **toàn bộ** kết quả (kể cả sample đã bị
loại) để phát hiện xu hướng, không chỉ đọc vài sample đầu file:
- Tỉ lệ dịch thành công / thất bại theo từng dataset.
- Phân loại lý do thất bại (còn tiếng Anh / tỉ lệ độ dài bất thường / JSON
  hỏng / lỗi API khác) — để biết loại lỗi nào phổ biến nhất.
- Phân bố tỉ lệ độ dài (ký tự VI / ký tự EN) trên các sample **dịch thành
  công**, đối chiếu với ngưỡng chấp nhận `[0.3, 2.0]` để xem ngưỡng đặt có
  hợp lý không (quá chặt sẽ loại oan sample dịch đúng nhưng ngắn/dài tự
  nhiên do khác biệt ngôn ngữ; quá lỏng sẽ lọt sample lỗi thật).
- Chi phí (token) và thời gian thực tế trên mỗi 1000 sample — xem mục 5."""

SECTION_3_NARRATIVE = """Đã đọc thủ công một số sample của cả 2 dataset (xem mục "10. Kiểm tra nhanh
vài sample đã dịch" và mục "13. Mẫu bản dịch để dán vào báo cáo" trong
notebook), nhận thấy:

- Không có sample nào còn sót tiếng Anh trong cả 6 mẫu bên dưới (3
  MusicBench, 3 MusicQA).
- **Thuật ngữ nhạc lý nhất quán giữa các lượt gọi độc lập**: cả 3 mẫu
  MusicBench đều dịch `riff`/`arpeggiated chord`/`hammer-on`/`slide`/`rim
  shot`/`double stop hammer-on` giống hệt nhau (giữ nguyên tiếng Anh), và
  `common time` → "nhịp 4/4", `E major` → "Mi trưởng (E major)" cũng lặp lại
  y hệt ở cả 3 mẫu — vì mỗi caption được dịch ở một lượt gọi Gemini riêng
  (khác batch), sự nhất quán này cho thấy lựa chọn thuật ngữ ổn định theo
  system prompt, không phải trùng hợp ở một mẫu.
- **Câu hỏi-câu trả lời của MusicQA giữ đúng ngữ cảnh với nhau** ở cả 3 mẫu:
  câu lệnh mô tả ngắn ("Describe the audio", "Describe the audio in detail")
  dịch đúng dạng câu lệnh khách quan ("Mô tả đoạn âm thanh này", "Mô tả chi
  tiết đoạn âm thanh"), còn câu hỏi trực tiếp ("What do you hear in the
  audio?") dịch đúng ngôi thứ 2 ("Bạn nghe thấy gì trong đoạn âm thanh?")
  thay vì nhầm sang câu lệnh — câu trả lời tương ứng luôn bám đúng nội dung
  được hỏi, không lệch chủ đề.
- **Cách xử lý thuật ngữ tiếng Anh nhất quán giữa 2 dataset**: tên thể loại
  nhạc (`alternative rock`, `post-rock`, `electronic`, `experimental`) giữ
  nguyên không dịch; từ mơ hồ được chú thích song ngữ kiểu
  `distorted` → "bị méo tiếng (distorted)", cùng kỹ thuật với
  "Mi trưởng (E major)" bên MusicBench.
- Độ dài và mức chi tiết bản dịch tương xứng bản gốc ở cả 6 mẫu, không thấy
  dấu hiệu cắt cụt hay hallucinate thêm nội dung không có trong câu gốc.

*(Nhận xét định tính này viết dựa trên lần đọc mẫu gần nhất -- nếu chất lượng
dịch thay đổi rõ rệt ở lần chạy mới, nên xem lại/viết lại đoạn này thủ công
trong file generate_report.py, hàm này không tự sinh lại phần nhận xét.)*"""

SECTION_4_LIMITATIONS = """## 4. Hạn chế đã biết

- Kiểm tra "còn tiếng Anh" ưu tiên tỉ lệ từ có dấu tiếng Việt (đáng tin hơn
  cho câu pha trộn thuật ngữ tiếng Anh có chủ đích) và chỉ rơi về `langdetect`
  khi câu hoàn toàn không dấu — vẫn có thể sai với câu trả lời rất ngắn,
  không dấu, không rõ ngôn ngữ (ví dụ chỉ có 1 từ như "Piano.").
- Không có bộ kiểm tra **định lượng** cho "câu hỏi-câu trả lời có khớp nghĩa
  hay không" sau khi dịch — chỉ phòng ngừa bằng cách dịch chung ngữ cảnh
  trong cùng 1 lượt gọi, chưa đo được mức độ khớp nghĩa bằng số.
- Ngưỡng tỉ lệ độ dài `[0.3, 2.0]` là ước lượng, nên đối chiếu lại với biểu
  đồ phân bố thực tế (mục 2) trên chính dataset này để tinh chỉnh nếu cần.
- Thống kê trong báo cáo này phụ thuộc vào `LIMIT`/số sample đã chạy tại thời
  điểm cập nhật — không phải kết quả trên toàn bộ 52.768 sample MusicBench +
  5.040 sample MusicQA trừ khi ghi rõ đã chạy full.
- Ước tính chi phí (mục 5) dùng giá tham khảo tự điền trong notebook, cần đối
  chiếu với bảng giá Vertex AI thật tại thời điểm chạy — không phải số chính
  thức từ Google Cloud Billing."""


def build_report_markdown(
    *,
    mb_quality: dict,
    mq_quality: dict,
    mb_merged: list[dict],
    mq_merged: list[dict],
    mb_cost_time: Optional[dict],
    mq_cost_time: Optional[dict],
    mb_extrap_10m: Optional[dict],
    mq_extrap_10m: Optional[dict],
    mb_failed: list[dict],
    mq_failed: list[dict],
    system_prompt: str,
    n_samples_for_report: int = 3,
) -> str:
    return "\n\n".join([
        REPORT_HEADER_AND_SECTION1,
        "## 2. Thống kê trên dữ liệu đã dịch\n\n"
        + render_stats_table(mb_quality, mq_quality)
        + "\n\n![Tỉ lệ dịch thành công theo dataset](translation_success_rate.png)"
          "\n\n![Phân bố tỉ lệ độ dài bản dịch](length_ratio_distribution.png)"
          "\n\n![Phân loại lý do dịch lỗi](failure_reasons_breakdown.png)",
        "## 3. Đánh giá định tính (đọc mẫu thủ công)\n\n"
        + SECTION_3_NARRATIVE + "\n\n"
        + render_sample_blocks(mb_merged, mq_merged, n_samples_for_report),
        SECTION_4_LIMITATIONS,
        "## 5. Chi phí & thời gian dịch (ước tính /1000 sample)\n\n"
        + render_cost_time_table(mb_cost_time, mq_cost_time)
        + "\n\n### Ước tính cho 10 triệu sample\n\n"
          "*(Ngoại suy tuyến tính từ bảng /1000 sample ở trên. Chỉ mang tính "
          "tham khảo thô, xem lưu ý ở mục 4.)*\n\n"
        + render_extrap_table(mb_extrap_10m, mq_extrap_10m),
        "## 6. Toàn bộ sample dịch lỗi\n\n"
        + render_failed_section("MusicBench", mb_failed) + "\n\n"
        + render_failed_section("MusicQA", mq_failed),
        "## 7. System prompt dùng để dịch\n\n"
        "System prompt gửi kèm mọi lệnh gọi Gemini (`SYSTEM_PROMPT` trong\n"
        "`translate_dataset.py`), áp dụng như nhau cho cả batch lẫn fallback từng\n"
        "sample riêng lẻ:\n\n"
        "```\n" + system_prompt.rstrip() + "\n```",
    ])


def save_report(save_path: Path, **kwargs) -> str:
    """Builds the report (see build_report_markdown for kwargs) and writes
    it to save_path. Returns the markdown text."""
    report_md = build_report_markdown(**kwargs)
    save_path.write_text(report_md, encoding="utf-8")
    return report_md
