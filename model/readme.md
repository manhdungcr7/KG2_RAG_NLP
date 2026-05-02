# Thư mục Model (Model Directory)

Thư mục này ban đầu được thiết kế để chứa các mô hình mã nguồn mở dùng cho dự án. Tuy nhiên, trong phiên bản hiện tại (sau quá trình tối ưu hóa), hệ thống đã được cấu hình để **tự động tải và quản lý mô hình** một cách thông minh hơn.

## 1. Mô hình Ngôn ngữ Lớn (LLM) và Embedding
Hệ thống sử dụng **Ollama** làm backend để chạy các mô hình nội bộ nhằm bảo mật dữ liệu và tiết kiệm tài nguyên.

**Các bước cài đặt:**
1. Cài đặt phần mềm [Ollama](https://ollama.com/) (Hỗ trợ Windows/macOS/Linux).
2. Tải mô hình sinh văn bản (LLM) đã được tối ưu hóa cho dự án (bản 1B):
   ```bash
   ollama run llama3.2:1b
   ```

Tải mô hình nhúng (Embedding Model) để xây dựng Vector Database:

```bash
ollama pull nomic-embed-text
```

Lưu ý: Mặc định Ollama sẽ lưu trữ các mô hình này trong ổ C (hoặc thư mục hệ thống), bạn không cần tải file vật lý vào thư mục `model/` này nữa.

## 2. Mô hình Xếp hạng lại (Reranker)
Hệ thống sử dụng mô hình `BAAI/bge-reranker-base` (nhẹ hơn bản large gốc nhưng vẫn đảm bảo độ chính xác).

Bạn KHÔNG cần tải mô hình này thủ công bằng lệnh `wget` nữa.

Khi bạn chạy script đánh giá RAG lần đầu tiên (ví dụ: `kg_rag_distractor.py`), thư viện `FlagEmbedding` và `transformers` (phiên bản `4.44.2`) sẽ tự động kết nối tới Hugging Face Hub, tải mô hình này về và lưu vào bộ nhớ cache của máy (thường là `~/.cache/huggingface/hub/`).
