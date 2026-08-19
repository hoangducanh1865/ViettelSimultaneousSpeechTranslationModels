# Đại từ nhân xưng rút gọn trong tiếng Việt (phương ngữ Nam Bộ)

Trong khẩu ngữ (đặc biệt tiếng Nam Bộ), người nói hay rút gọn **"[danh xưng] + ấy"**
thành một âm tiết duy nhất, đổi thanh điệu, để chỉ **ngôi thứ 3** (she/he/her/him) —
khác hẳn danh xưng gốc, vốn thường dùng để xưng hô trực tiếp (**ngôi thứ 2**, "you").

| Rút gọn (ngôi 3) | Nghĩa đầy đủ    | Danh xưng gốc (ngôi 2, "you") |
|---|---|---|
| chỉ  | chị ấy (she/her) | chị |
| ảnh  | anh ấy (he/him)  | anh |
| ổng  | ông ấy (he/him)  | ông |
| bả   | bà ấy (she/her)  | bà  |
| cổ   | cô ấy (she/her)  | cô  |
| chả  | chú ấy (he/him)  | chú |

## Vì sao quan trọng

ASR đôi khi nhận dạng đúng dạng rút gọn này (ví dụ "chỉ"), nhưng nếu bước hậu xử lý
coi đó là lỗi chính tả rồi "sửa" về danh xưng gốc ("chị"), sẽ **xoá mất tín hiệu
ngôi thứ 3** — khiến bước dịch sau đó hiểu nhầm thành ngôi thứ 2 ("you") thay vì
đúng là "she/he/her/him" nói về một người thứ 3 ngoài cuộc hội thoại.

## Quy tắc áp dụng

- **Không bao giờ** "sửa" các từ rút gọn này về lại danh xưng gốc hay bất kỳ dạng
  nào khác — kể cả khi làm sạch (cleanup) văn bản ASR. Giữ nguyên y hệt ("chỉ" ở
  văn bản tiếng Việt vẫn phải là "chỉ"). Đó không phải lỗi chính tả, đó là ngữ
  pháp khẩu ngữ hợp lệ, và ý nghĩa ngôi thứ 3 phải được **suy luận đúng ở bước
  dịch** dựa trên bảng tra cứu này + ngữ cảnh câu, không phải bằng cách viết lại
  văn bản gốc.
- Khi dịch sang tiếng Anh: các từ rút gọn này luôn dịch là "she/her" hoặc
  "he/him" (ngôi thứ 3), **không bao giờ** dịch là "you".
- Phân biệt với danh xưng trần (không mang nghĩa "ấy" ẩn): "chị", "anh", "ông",
  "bà", "cô" đứng một mình có thể là ngôi thứ 2 (xưng hô trực tiếp) hoặc chủ ngữ
  được nêu tên lần đầu trong câu chuyện (trước khi chuyển sang dùng dạng rút
  gọn cho các lần nhắc sau) — cần dựa vào ngữ cảnh cả câu để xác định, không
  suy diễn máy móc.
