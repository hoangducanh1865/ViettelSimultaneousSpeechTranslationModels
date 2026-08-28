# ASR Labeling App

Web app nội bộ để nghe audio, xem 3 bản ASR ứng viên (Google STT / ASR nội bộ Viettel /
PhoWhisper) sinh ra từ `noteboooks/TestSet_construction.ipynb`, và nhập bản ASR final đã
sửa tay. Kết quả xuất ra CSV khớp đúng format `asr_manual_check.csv` để thả thẳng vào
Drive cho GATE 1 của notebook đọc tiếp.

## Kiến trúc & vì sao chọn vậy

- **Audio**: nằm ngay trong repo, ở `backend/audio_data/`, được backend FastAPI serve
  trực tiếp qua endpoint `/audio/{filename}?key=...` (chặn bằng 1 secret key đơn giản,
  `AUDIO_ACCESS_KEY`). **Không** đẩy audio lên GitHub public + CDN (jsDelivr) dù file
  nhỏ -- VieSpeaker/VietnamCeleb/VoxVietnam là dataset nghiên cứu có điều khoản sử dụng
  hạn chế, đưa audio lên nơi công khai có thể vi phạm điều khoản đó. Giữ repo/deploy ở
  chế độ private + gate bằng key là đủ cho nhu cầu 1-2 người review nội bộ.
- **Label (final_asr_text)**: lưu trong Postgres (add-on của Railway), không dùng
  SQLite -- ổ đĩa container của Railway là ephemeral, redeploy sẽ mất dữ liệu SQLite.
- **Đồng bộ lại Drive**: endpoint `/api/export/asr-check.csv` xuất đúng schema
  `asr_manual_check.csv` -- tải về, kéo thả đè vào thư mục `test_set_construction/` trên
  Drive, chạy lại GATE 1 trong notebook như bình thường, không cần convert gì thêm.

## Luồng dữ liệu đầy đủ

```
Colab (manifest.jsonl + audio trên Drive)
    │  prepare_data.py
    ▼
backend/audio_data/*.wav + backend/seed_samples.json   (commit vào git)
    │  git push
    ▼
Railway (backend) ── seed.py nạp vào Postgres
    │  REST API
    ▼
Vercel (frontend) ── người review nghe + nhập final_asr_text
    │  export CSV
    ▼
Thả CSV vào Drive ── GATE 1 trong TestSet_construction.ipynb đọc tiếp
```

## 1. Chuẩn bị dữ liệu (chạy trong Colab, nơi có Drive + manifest.jsonl)

```python
!python prepare_data.py \
    --manifest /content/drive/MyDrive/.../test_set_construction/manifest.jsonl \
    --audio-out ./audio_data \
    --seed-out ./seed_samples.json
```

Zip `audio_data/` + `seed_samples.json`, tải về máy, giải nén đè vào:
- `src/labeling_app/backend/audio_data/`
- `src/labeling_app/backend/seed_samples.json`

rồi `git add`/`commit`/`push` như bình thường (audio nhỏ nên push git bình thường, không
cần Git LFS ở quy mô vài chục-vài trăm file ngắn).

## 2. Deploy backend lên Railway

1. Tạo project mới trên Railway, connect vào repo GitHub này, set **root directory** =
   `src/labeling_app/backend`.
2. Add **Postgres** add-on trong cùng project -- Railway tự inject biến `DATABASE_URL`.
3. Set thêm biến môi trường (xem `.env.example`):
   - `AUDIO_ACCESS_KEY` -- chuỗi bí mật tự chọn.
   - `ALLOWED_ORIGINS` -- domain Vercel của bạn (thêm sau khi deploy frontend xong).
4. Railway tự nhận `Procfile` (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`).
5. Sau khi deploy xong, chạy 1 lần để nạp dữ liệu vào Postgres:
   ```
   railway run python seed.py
   ```
   (chạy lại lệnh này bất cứ khi nào bạn thêm sample mới vào `seed_samples.json` --
   idempotent, không đè `final_asr_text` của sample đã submit).

## 3. Deploy frontend lên Vercel

1. Import repo, set **root directory** = `src/labeling_app/frontend`.
2. Set biến môi trường `NEXT_PUBLIC_API_URL` = URL backend Railway (vd
   `https://xxx.up.railway.app`).
3. Deploy. Quay lại Railway, cập nhật `ALLOWED_ORIGINS` thành domain Vercel vừa có,
   redeploy backend.

## 4. Chạy local để test trước khi deploy

Backend:
```bash
cd backend
pip install -r requirements.txt
python seed.py          # cần seed_samples.json + audio_data/ đã có sẵn
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

## Tính năng UI

- Sidebar chọn sample bất kỳ (chấm xanh = đã submit, chấm xám = chưa).
- Nút Back/Next chuyển sample tuần tự.
- Audio player nghe trực tiếp.
- 3 khối hiển thị bản ASR Google/nội bộ/PhoWhisper -- bấm vào 1 khối để copy nội dung
  đó vào ô sửa bên dưới (tránh phải gõ lại từ đầu nếu 1 bản đã đúng phần lớn).
- Ô textarea sửa tay bản ASR final, nút **Submit** (lưu + tự nhảy sang sample tiếp theo)
  và nút **Làm lại** (xoá bản đã submit, quay về trạng thái pending để sửa lại từ đầu).
- Link tải CSV kết quả bất cứ lúc nào (không cần đợi làm xong hết mới tải được).
