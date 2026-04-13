# Nguyen Vo Dang Khoa Architects RAG Chatbot

Hybrid RAG chatbot cho dữ liệu công ty kiến trúc/nội thất, gồm:

- FastAPI backend
- Streamlit UI
- Qdrant vector database
- Hybrid retrieval (dense + BM25-style sparse)
- Cross-encoder reranking
- Local LLM generation

## Những gì đã được chỉnh trong repo này

- Sửa lệch answer trước/sau rerank: answer grounded, context LLM và `sources` giờ đều bám cùng tập document sau rerank.
- Đồng bộ contract backend/UI cho `sources`, `debug`, `latency`, `request_id`.
- Đồng bộ config LLM giữa `settings.yaml`, `env.example` và runtime env override.
- Thêm logging + latency theo request/stage.
- Backend chuyển sang stateless; rate-limit in-memory chỉ còn là chế độ demo tùy chọn.
- Thêm Docker để chạy Qdrant + API + UI.
- Thêm bộ evaluation 40 câu và script chấm end-to-end.

## Kiến trúc nhanh

```text
processed JSON data
  -> chunking theo entity
  -> embedding dense + sparse
  -> Qdrant hybrid index
  -> hybrid retrieve
  -> rerank
  -> grounded answer / LLM answer
  -> FastAPI
  -> Streamlit UI
```

## Yêu cầu

- Python 3.10+
- Docker Desktop hoặc Docker Engine + Compose
- RAM đủ để chạy embedding/reranker/LLM bạn chọn

## Cấu hình

Repo dùng `src/config/settings.yaml` làm default và cho phép override bằng env vars.

File mẫu: `env.example`

Khi chạy local, các entrypoint chính như API, UI và pipeline sẽ tự động load file `.env` ở thư mục root của project bằng `python-dotenv`.
Biến môi trường đã export sẵn trong shell vẫn giữ ưu tiên cao hơn `.env`.

Các biến quan trọng:

- `QDRANT_URL`: URL Qdrant
- `RAW_DATA_FILE`: đường dẫn file raw JSON cho `python data/load_data.py`
- `LLM_PROVIDER`: `huggingface_local` hoặc `ollama`
- `LLM_MODEL_NAME`: tên model theo provider
- `LLM_BASE_URL`: dùng cho Ollama
- `LLM_DEVICE`: `cpu`, `cuda`, `auto`
- `LLM_ENABLE_FALLBACK`: `false` mặc định; nếu `true` thì mới cho phép route chat dùng LLM fallback khi grounded answer không đủ
- `EMBEDDING_DEVICE`: `cpu` hoặc `cuda`
- `RERANKING_DEVICE`: `cpu` hoặc `cuda`
- `DEMO_RATE_LIMIT_ENABLED`: `false` mặc định; nếu bật thì chỉ là rate-limit in-memory cho demo local

## Chạy Local End-to-End trên Windows PowerShell

Các bước dưới đây giả định bạn đang đứng ở thư mục root của repo sau khi `git clone`.

### 1. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2. Kích hoạt môi trường ảo

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Cài dependency

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Copy file `.env`

```powershell
Copy-Item env.example .env
```

Sau khi copy, các lệnh `python data/load_data.py`, `python -m src.pipeline`,
`uvicorn src.api.routes.app:app ...` và `python -m streamlit run src/ui/chatbot.py`
sẽ tự đọc `.env` mà không cần `set` hay `export` thủ công trước.

Nếu UI cần trỏ sang API khác, bạn có thể thêm vào `.env`:

```env
API_BASE_URL=http://127.0.0.1:8000
```

Nếu bạn dùng `huggingface_local`, giữ mặc định hoặc chỉnh `LLM_MODEL_NAME` / `LLM_DEVICE`.

Mặc định hệ thống đang chạy theo hướng `grounded-only`: nếu không có câu trả lời đủ chắc từ dữ liệu đã retrieve/rerank thì API sẽ không tự động fallback sang LLM.

Nếu bạn dùng Ollama, đổi tối thiểu:

```env
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:3b
LLM_BASE_URL=http://localhost:11434
```

### 5. Chạy Qdrant

```powershell
docker compose up -d qdrant
```

### 6. Load data raw sang `data/processed`

```powershell
python data/load_data.py
```

Script này hỗ trợ 3 cách chọn file raw theo thứ tự ưu tiên:

1. `--input <path>`
2. Biến môi trường `RAW_DATA_FILE` trong shell hoặc `.env`
3. File mặc định cũ

File mặc định hiện tại là:

- `data/raw/database_export_nguyen_vo_dang_khoa_completed.json`

Repo hiện đã có sẵn file raw mẫu này, nên sau khi clone bạn có thể chạy ngay `python data/load_data.py`.

Ví dụ dùng file mặc định:

```powershell
python data/load_data.py
```

Ví dụ truyền file trực tiếp:

```powershell
python data/load_data.py --input data/raw/database_export_nguyen_vo_dang_khoa_completed.json
```

Ví dụ dùng `.env`:

```env
RAW_DATA_FILE=data/raw/database_export_nguyen_vo_dang_khoa_completed.json
```

Sau đó chạy:

```powershell
python data/load_data.py
```

Script sẽ log rõ chính xác file raw nào đang được dùng. Nếu path không tồn tại, script sẽ báo lỗi dễ hiểu và thoát với mã lỗi khác `0`.

### 7. Chạy pipeline để build / update index an toàn

```powershell
python -m src.pipeline
```

Lệnh mặc định sẽ `upsert` vào collection hiện có và không recreate collection cũ.
Nếu bạn cần full rebuild có chủ đích, dùng:

```powershell
python -m src.pipeline --recreate-collection
```

### 8. Chạy API

```powershell
uvicorn src.api.routes.app:app --host 0.0.0.0 --port 8000 --reload
```

### 9. Chạy UI

Mở một PowerShell khác, kích hoạt lại `.venv`, rồi chạy:

```powershell
python -m streamlit run src/ui/chatbot.py
```

### 10. Mở ứng dụng

- UI: `http://localhost:8501`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/readiness`

## Lỗi thường gặp

### 1. Streamlit launcher path issue

Nếu `streamlit run ...` báo lỗi không tìm thấy launcher hoặc dùng nhầm Python, hãy luôn chạy theo dạng:

```powershell
python -m streamlit run src/ui/chatbot.py
```

Cách này dùng đúng interpreter của `.venv` đang active và ổn định hơn trên Windows.

### 2. `.env` không hoạt động

Kiểm tra các điểm sau:

- File phải nằm đúng ở root của repo: `.env`
- Sau khi sửa `.env`, hãy chạy lại pipeline/API/UI để nạp cấu hình mới
- Biến môi trường đã export sẵn trong PowerShell sẽ ưu tiên hơn `.env`

Kiểm tra nhanh:

```powershell
@'
import os
from src.core.setting_loader import ensure_env_loaded

ensure_env_loaded()
print(os.getenv("QDRANT_URL"))
'@ | python -
```

### 3. Qdrant chưa chạy

Nếu pipeline hoặc API báo lỗi không kết nối được Qdrant, hãy khởi động lại:

```powershell
docker compose up -d qdrant
```

Bạn có thể kiểm tra nhanh:

- `http://localhost:6333/dashboard`
- `http://localhost:8000/health`

## Chạy Bằng Docker

### 1. Build image

```powershell
docker compose build
```

### 2. Khởi động Qdrant

```powershell
docker compose up -d qdrant
```

### 3. Build index từ container API

```powershell
docker compose run --rm api python -m src.pipeline
```

Nếu cần full rebuild collection trong Docker:

```powershell
docker compose run --rm api python -m src.pipeline --recreate-collection
```

### 4. Khởi động API + UI

```powershell
docker compose up -d api ui
```

### 5. Truy cập

- UI: `http://localhost:8501`
- API: `http://localhost:8000`

Ghi chú với Ollama trong Docker:

- Docker Desktop: có thể dùng `LLM_BASE_URL=http://host.docker.internal:11434`
- Linux: đổi `LLM_BASE_URL` sang IP host hoặc reverse proxy phù hợp

## API Contract

### Request

```http
POST /api/chat
Content-Type: application/json

{
  "query": "Hotline công ty là gì?",
  "debug": true,
  "top_k": 10,
  "session_id": "optional-client-session-id"
}
```

### Response

```json
{
  "answer": "Hotline của Nguyen Vo Dang Khoa Architects là: 0909.268.416",
  "session_id": "...",
  "request_id": "...",
  "latency": {
    "retrieval_ms": 52.4,
    "rerank_ms": 31.8,
    "answer_ms": 2.1,
    "total_ms": 89.7
  },
  "sources": [
    {
      "index": 1,
      "title": "Nguyen Vo Dang Khoa Architects",
      "source_type": "company_info",
      "chunk_type": "contact_info",
      "doc_id": "...",
      "snippet": "...",
      "scores": {
        "hybrid": 0.123,
        "dense": 0.987,
        "bm25": 3.214,
        "rerank": 7.654
      }
    }
  ],
  "debug": []
}
```

## Evaluation

Repo có sẵn bộ eval 40 câu trong `evaluation/eval_cases.py`.

### Validate dataset

```powershell
python evaluation/run_eval.py --validate-only
```

### Chạy eval với API local

```powershell
python evaluation/run_eval.py --api-base http://127.0.0.1:8000 --output evaluation/report.json
```

### Chạy nhanh vài câu đầu

```powershell
python evaluation/run_eval.py --api-base http://127.0.0.1:8000 --limit 10
```

## Logging và Latency

- Log file: `logs/application.log`
- Có `request_id` và tổng thời gian xử lý ở middleware API
- Chat endpoint trả breakdown `retrieval_ms`, `rerank_ms`, `answer_ms`, `total_ms`
- Có thể đổi mức log bằng `LOG_LEVEL`

## Demo-Only Note

Backend hiện stateless ở phía server.

- Không còn lưu lịch sử chat in-memory để phục vụ logic trả lời.
- `session_id` chỉ dùng để client/UI tự gắn cuộc hội thoại cho dễ theo dõi.
- `DEMO_RATE_LIMIT_ENABLED=true` chỉ nên dùng khi demo local vì đây là rate-limit in-memory, không phù hợp production hay multi-instance.

## Một vài câu hỏi mẫu

- `Cho tôi thông tin công ty`
- `Hotline công ty là gì?`
- `Công ty có những phong cách nội thất nào?`
- `Dự án nào ở Bình Tân?`
- `Cho tôi biết về dự án Căn hộ gia đình Phúc Đạt mẫu Japandi`
- `Có bài viết nào về xu hướng thiết kế nhà phố hiện đại 2026 không?`

## Cấu trúc thư mục chính

```text
src/
  api/routes/         FastAPI routes
  core/               schema, settings, logging, startup
  llm/                prompt + generation + grounded answer
  rag/                chunking, retrieval, reranking, vector store
  ui/                 Streamlit app

evaluation/
  eval_cases.py       40 câu eval
  run_eval.py         script chấm end-to-end
```
