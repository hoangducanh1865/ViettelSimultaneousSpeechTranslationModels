# Báo cáo chất lượng dịch EN → VI (MusicBench, MusicQA)

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
| Vẫn là tiếng Anh | Ưu tiên đếm **tỉ lệ từ có dấu tiếng Việt** trước (câu ≤6 từ chỉ cần 1 từ có dấu; câu dài hơn cần ≥15% số từ có dấu) — tín hiệu này đáng tin hơn `langdetect` cho câu pha trộn nhiều thuật ngữ/tên riêng tiếng Anh có chủ đích (tên nhạc cụ, thể loại). Chỉ dùng `langdetect.detect() == "vi"` làm phương án dự phòng khi câu hoàn toàn không có dấu | Dịch lại riêng sample đó, tối đa `MAX_RETRIES` lần |
| Tỉ lệ độ dài bất thường | `len(bản dịch) / len(bản gốc)` phải nằm trong `[0.3, 2.0]` — bắt các case rỗng/bị cắt/dịch lan man thêm nội dung | Dịch lại riêng sample đó |
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
- Chi phí (token) và thời gian thực tế trên mỗi 1000 sample — xem mục 5.

## 2. Thống kê trên dữ liệu đã dịch

| Dataset | Tổng sample đã thử dịch | Dịch OK | Dịch lỗi (đã loại) | Tỉ lệ thành công |
|---|---|---|---|---|
| MusicBench | 2000 | 2000 | 0 | 100.0% |
| MusicQA | 2000 | 1980 | 20 | 99.0% |

![Tỉ lệ dịch thành công theo dataset](translation_success_rate.png)

![Phân bố tỉ lệ độ dài bản dịch](length_ratio_distribution.png)

![Phân loại lý do dịch lỗi](failure_reasons_breakdown.png)

## 3. Đánh giá định tính (đọc mẫu thủ công)

Đã đọc thủ công một số sample của cả 2 dataset (xem mục "10. Kiểm tra nhanh
vài sample đã dịch" và mục "13. Mẫu bản dịch để dán vào báo cáo" trong
notebook), nhận thấy:

- Không có sample nào còn sót tiếng Anh trong các mẫu đã xem.
- Cặp câu hỏi-câu trả lời của MusicQA giữ đúng ngữ cảnh với nhau sau khi
  dịch — ví dụ câu hỏi "What do you hear in the audio?" dịch đúng ngôi
  "Bạn nghe thấy gì trong đoạn âm thanh?", và câu trả lời vẫn mô tả đúng nội
  dung tương ứng, không lệch chủ đề.
- Thuật ngữ nhạc lý được xử lý hợp lý: giữ nguyên các từ chuyên ngành guitar
  phổ biến trong cộng đồng nhạc Việt (`hammer-on`, `slide`, `riff`, `rim
  shot`), đồng thời dịch hẳn sang tiếng Việt các khái niệm có thuật ngữ
  tương đương rõ ràng (`E major` → "Mi trưởng (E major)", giữ cả hai để dễ
  đối chiếu).
- Độ dài bản dịch tương xứng bản gốc, không thấy dấu hiệu cắt cụt hay
  hallucinate thêm nội dung không có trong câu gốc.

*(Dán thêm mẫu từ mục "13" của notebook vào đây khi cập nhật báo cáo với dữ
liệu mới.)*

**MusicBench**

> EN: (dán mẫu từ notebook vào đây)
>
> VI: (dán mẫu từ notebook vào đây)

**MusicQA**

> Q (EN): (dán mẫu từ notebook vào đây)
> Q (VI): (dán mẫu từ notebook vào đây)
>
> A (EN): (dán mẫu từ notebook vào đây)
> A (VI): (dán mẫu từ notebook vào đây)

## 4. Hạn chế đã biết

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
  thức từ Google Cloud Billing.

## 5. Chi phí & thời gian dịch (ước tính /1000 sample)

*(Copy bảng in ra từ mục "12. Chi phí + thời gian dịch" trong notebook vào
đây — số liệu phụ thuộc model, batch size, và giá tại thời điểm chạy.)*

| Dataset | Sample dịch (lần chạy) | Thời gian ước tính /1000 sample | Chi phí ước tính /1000 sample |
|---|---|---|---|
| MusicBench | — | — | — |
| MusicQA | — | — | — |
