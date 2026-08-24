# Báo cáo chất lượng dịch EN → VI (MusicBench, MusicQA)

Báo cáo mô tả cách đánh giá chất lượng bản dịch tự động (Gemini/Vertex AI, qua
`translate_dataset.py`) cho 2 dataset `MusicBench` (caption nhạc) và `MusicQA`
(hỏi-đáp về nhạc), và thống kê kết quả trên dữ liệu đã dịch. Code sinh số liệu
+ biểu đồ: mục "## 11. Đánh giá chất lượng dịch" trong
`noteboooks/Translate_QA_Datasets.ipynb`.

## 1. Tiêu chí đánh giá

Chất lượng được kiểm tra ở 2 lớp: **lúc dịch** (tự động loại sample lỗi ngay
khi phát hiện, không đợi hậu kiểm) và **hậu kiểm** (thống kê lại toàn bộ kết
quả sau khi chạy xong để phát hiện xu hướng lỗi hệ thống nếu có).

### 1.1 Kiểm tra lúc dịch (`translate_dataset.py`)

| Tiêu chí | Cách kiểm tra | Xử lý khi fail |
|---|---|---|
| Vẫn là tiếng Anh | `langdetect.detect() == "vi"`; fallback nếu langdetect không phân loại được: đếm ký tự có dấu tiếng Việt | Dịch lại riêng sample đó, tối đa `MAX_RETRIES` lần |
| Tỉ lệ độ dài bất thường | `len(bản dịch) / len(bản gốc)` phải nằm trong `[0.3, 3.5]` — bắt các case rỗng/bị cắt/dịch lan man thêm nội dung | Dịch lại riêng sample đó |
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
  công**, đối chiếu với ngưỡng chấp nhận `[0.3, 3.5]` để xem ngưỡng đặt có
  hợp lý không (quá chặt sẽ loại oan sample dịch đúng nhưng ngắn/dài tự
  nhiên do khác biệt ngôn ngữ; quá lỏng sẽ lọt sample lỗi thật).

## 2. Thống kê trên dữ liệu đã dịch

*(Chạy mục "## 11" trong notebook, copy bảng in ra từ cell cuối vào đây thay
cho bảng mẫu bên dưới — số liệu phụ thuộc `LIMIT`/lần chạy cụ thể.)*

| Dataset | Tổng sample đã thử dịch | Dịch OK | Dịch lỗi (đã loại) | Tỉ lệ thành công |
|---|---|---|---|---|
| MusicBench | 2000 | 2000 | 0 | 100.0% |
| MusicQA | 2000 | 1980 | 20 | 99.0% |

![Tỉ lệ dịch thành công theo dataset](translation_success_rate.png)

![Phân bố tỉ lệ độ dài bản dịch](length_ratio_distribution.png)

![Phân loại lý do dịch lỗi](failure_reasons_breakdown.png)

## 3. Đánh giá định tính (đọc mẫu thủ công)

Đã đọc thủ công một số sample của cả 2 dataset (xem mục "10. Kiểm tra nhanh
vài sample đã dịch" trong notebook), nhận thấy:

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

## 4. Hạn chế đã biết

- Kiểm tra "còn tiếng Anh" dựa vào `langdetect` + đếm ký tự có dấu — có thể
  bỏ sót câu dịch rất ngắn, hoặc câu pha trộn nhiều tên riêng/thuật ngữ tiếng
  Anh giữ nguyên có chủ đích (tên nhạc cụ, thể loại nhạc), khiến bộ đếm dấu
  tiếng Việt không đủ tin cậy cho câu ngắn.
- Không có bộ kiểm tra **định lượng** cho "câu hỏi-câu trả lời có khớp nghĩa
  hay không" sau khi dịch — chỉ phòng ngừa bằng cách dịch chung ngữ cảnh
  trong cùng 1 lượt gọi, chưa đo được mức độ khớp nghĩa bằng số.
- Ngưỡng tỉ lệ độ dài `[0.3, 3.5]` là ước lượng ban đầu (kinh nghiệm từ pha
  dịch VI→EN của pipeline ASR khác trong repo) — nên đối chiếu lại với biểu
  đồ phân bố thực tế (mục 2) trên chính dataset này để tinh chỉnh nếu cần.
- Thống kê trong báo cáo này phụ thuộc vào `LIMIT`/số sample đã chạy tại thời
  điểm cập nhật — không phải kết quả trên toàn bộ 52.768 sample MusicBench +
  5.040 sample MusicQA trừ khi ghi rõ đã chạy full.
