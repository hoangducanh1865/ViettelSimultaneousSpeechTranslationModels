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

- **Khi làm sạch (cleanup) văn bản ASR: TUYỆT ĐỐI không đổi các từ này theo bất
  kỳ hướng nào.**
  - Không đổi dạng rút gọn về lại danh xưng gốc ("chỉ" -> "chị"): đó không phải
    lỗi chính tả, đó là ngữ pháp khẩu ngữ hợp lệ.
  - **Cũng không được tự suy đoán rồi đổi danh xưng gốc thành dạng rút gọn**
    ("ông" -> "ổng") chỉ vì đoán là ngôi thứ 3 — chỉ vì cả đoạn/câu đang nói về
    người thứ 3 không có nghĩa ASR "phải" nhận dạng ra dạng rút gọn; nếu ASR
    nhận dạng ra "ông" thì giữ nguyên "ông", không tự ý viết thành "ổng".
  - Nói ngắn gọn: chữ nào ASR nhận dạng ra, giữ nguyên y hệt chữ đó. Việc suy
    luận ngôi thứ 2 hay ngôi thứ 3 là việc của **bước dịch**, không phải việc
    của cleanup.
- Khi dịch sang tiếng Anh: dựa vào từ mà ASR *thực sự* nhận dạng được (không tự
  suy đoán từ ASR không có) + ngữ cảnh cả câu để xác định ngôi. Dạng rút gọn
  (chỉ, ảnh, ổng, bả, cổ, chả) luôn là "she/her"/"he/him" (ngôi thứ 3), không
  bao giờ là "you". Nếu trong cùng một đoạn có cả danh xưng gốc và dạng rút gọn
  cùng chỉ về một người (ví dụ câu trước dùng "chị" để giới thiệu, câu sau dùng
  "chỉ" khi nhắc lại), hãy suy luận rằng cả hai đều chỉ cùng một người đó (ngôi
  thứ 3) dựa trên mạch văn, rồi dịch nhất quán — nhưng đây là suy luận khi dịch,
  KHÔNG phải lý do để viết lại văn bản tiếng Việt gốc.
- Danh xưng gốc đứng một mình, không có dạng rút gọn nào khác trong cùng đoạn
  văn làm bằng chứng, có thể là ngôi thứ 2 (xưng hô trực tiếp) hoặc ngôi thứ 3
  (chủ ngữ được nêu tên, văn phong trang trọng/văn viết) — dựa vào ngữ cảnh cả
  câu để xác định, không suy diễn máy móc chỉ vì "nghe có vẻ" là ngôi thứ 3.
