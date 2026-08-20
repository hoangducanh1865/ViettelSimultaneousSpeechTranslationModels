Các bước thực hiện

Dataset: mỗi sample = (audio tiếng Anh do người Việt nói, text tiếng Anh, text tiếng Việt),
lấy trực tiếp từ phụ đề crawl được trên YouTube -- không qua ASR/LLM. Xem
`noteboooks/Vietnamese_people_speak_English_dataset.ipynb` để chạy toàn bộ trên Colab.

Bước 1: Thêm danh sách video cần tải

Tạo file `urls.txt` trong thư mục này và dán các đường link YouTube (video hoặc
playlist, mỗi link 1 dòng) -- `crawl.py` truyền thẳng cho yt-dlp nên playlist sẽ
tự được mở rộng.

Bước 2: Crawl (tải Audio + Phụ đề từ YouTube)

```bash
python crawl.py --urls-file urls.txt --output-dir dataset/raw_audio
```

Mặc định không cần cookie -- chỉ cần khi YouTube trả về `429 Too Many Requests`
(thường gặp hơn khi chạy trên Colab do dùng chung IP):

> **Lấy cookie YouTube:** Edge/Chrome bản mới mã hoá cookie theo cách yt-dlp không tự giải mã được (`--cookies-from-browser` sẽ lỗi `Failed to decrypt with DPAPI`), nên cần xuất cookie thủ công:
> 1. Cài extension **"Get cookies.txt LOCALLY"** vào Edge/Chrome.
> 2. Đăng nhập YouTube, mở youtube.com, bấm icon extension → **Export** (định dạng Netscape) → lưu thành `cookies.txt`.
> 3. `cookies.txt` chứa thông tin đăng nhập, đã được thêm vào `.gitignore`, **không commit lên git**. Trên Colab, upload lên Drive rồi trỏ `--cookies` vào đường dẫn đó.

```bash
python crawl.py --urls-file urls.txt --output-dir dataset/raw_audio --cookies cookies.txt
```

Cài thêm hỗ trợ chống bị chặn (chỉ cần làm 1 lần, trên máy local):

```powershell
pip install "yt-dlp[default,curl-cffi]"
winget install DenoLand.Deno
```

> - `--cookies`: dùng cookie đăng nhập YouTube đã xuất thủ công, giúp giảm mạnh khả năng bị chặn `429`.
> - `crawl.py` luôn bật `--js-runtimes deno` (bắt buộc để yt-dlp giải mã signature YouTube, tránh lỗi `HTTP Error 403: Forbidden`) và các tham số `--sleep-*` để giãn cách request.
> - Nếu IP đã bị khoá tạm thời từ lần chạy trước, có thể vẫn cần chờ vài chục phút trước khi thử lại dù đã thêm cookie.

Bước 3: Cắt segment + ghép ground truth + lọc chất lượng

```bash
python process_videos.py --raw-dir dataset/raw_audio --out-dir dataset/final
```

Với mỗi video: cắt audio theo mốc thời gian của phụ đề tiếng Anh (gộp các cue
liền nhau tới một độ dài mục tiêu), lấy phụ đề tiếng Việt đè cùng khoảng thời
gian làm ground truth tiếng Việt, rồi lọc lại bằng `lang_id.py` (đảm bảo đoạn
audio thực sự là tiếng Anh). Ghi `manifest.jsonl` + `build_report.json` vào
`--out-dir`; chạy lại lệnh trên sẽ tự bỏ qua video đã xử lý xong (resume-safe).
Dùng `--limit N` để chạy thử trên vài video trước.

# Thư viện

yt-dlp: tải audio + phụ đề từ YouTube (`crawl.py`).

webvtt-py: đọc/phân tích phụ đề `.vtt`/`.srt` kèm mốc thời gian (`process_videos.py`).

soundfile, transformers, torch, torchaudio: dùng bởi `lang_id.py` (bước lọc
ngôn ngữ, đã có sẵn trong `src/test_set/public/`, không cần cài thêm ngoài
`requirements.txt` gốc của repo).
