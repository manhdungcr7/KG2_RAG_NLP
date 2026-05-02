Markdown
# KG²RAG: Knowledge Graph Retrieval-Augmented Generation

Dự án này là bản triển khai thực nghiệm hệ thống **KG²RAG**, nhằm giải quyết bài toán Multi-hop Question Answering bằng cách kết hợp Đồ thị tri thức (Knowledge Graph) và Retrieval-Augmented Generation (RAG).

## 🚀 Tiến độ hiện tại (Current Progress)

Hệ thống đã hoàn thiện toàn bộ pipeline từ tiền xử lý (Offline Preprocessing) đến bước đánh giá (Evaluation) trên mô hình nội bộ.

### 1. Offline Processing (Đã hoàn thành)
Đã chạy thành công thuật toán trích xuất bộ ba (Triplets) để xây dựng sub-KGs cho 4 bộ datasets lớn:
- [x] **HotpotQA**: Xây dựng thành công đồ thị tri thức cho tập distractor.
- [x] **MuSiQue**: Tối ưu hóa pipeline (chống văng VRAM) và lưu trữ dạng Dictionary KGs.
- [x] **TriviaQA**: Tích hợp cơ chế Resume (tự động khôi phục khi ngắt quãng), chống lỗi ký tự file trên Windows.
- [x] **2WikiMultihopQA (Dataset mở rộng)**: Chuyển đổi thành công cấu trúc `.parquet`, trích xuất hoàn thiện ~55,000 thực thể.

### 2. Evaluation Pipeline (Đã giải quyết)
Đã khắc phục hoàn toàn các điểm nghẽn kỹ thuật trong mã nguồn gốc:
- Xử lý triệt để lỗi xung đột thư viện Hugging Face (`transformers==4.44.2` vs `FlagEmbedding`).
- Tích hợp giới hạn Context Window (`num_ctx: 4096`) cho `llama3.2:1b` để tránh tràn VRAM (lỗi 13.1 GiB).
- Hệ thống nhúng (`nomic-embed-text`) và xếp hạng (`bge-reranker-base`) đã hoạt động ổn định trong luồng truy vấn.
- Đã chạy mượt mà script `kg_rag_distractor.py` trên tập HotpotQA.

---

## 🛠 Hướng dẫn Cài đặt & Khởi chạy

### Cài đặt môi trường
Vì source code gốc có nhiều thư viện "đụng nhau", vui lòng cài đặt chính xác các phiên bản trong `requirements.txt`:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
(Lưu ý: Data đã được gỡ khỏi repo để tránh quá tải. Vui lòng tải 4 bộ datasets và đặt vào thư mục data/ theo đúng cấu trúc mã nguồn).
```

Lệnh chạy Thực nghiệm (Ví dụ với HotpotQA)
```bash
cd code
python kg_rag_distractor.py --dataset hotpotqa --data_path ../data/hotpotqa/hotpot_dev_distractor_v1.json --kg_dir ../data/hotpotqa/kgs/extract_subkgs --result_path ../output/hotpot/hotpot_dev_distractor_v1_kgrag.json
```

🎯 Định hướng các bước tiếp theo (Next Steps)
Để hoàn thiện đồ án này một cách xuất sắc nhất, các tác vụ sau sẽ được tiến hành:

Hoàn thiện Chạy Đánh Giá (Evaluation Run):

Chạy script kg_rag_distractor.py cho 3 bộ dataset còn lại (MuSiQue, TriviaQA, 2Wiki).

Khuyến nghị: Nên cắt nhỏ dữ liệu (vd: 500 câu/dataset) để kịp thời gian thực nghiệm đồ án.

Chấm điểm thuật toán (Metrics Calculation):

Viết hoặc sử dụng script đối chiếu kết quả từ thư mục output/ với Ground Truth (đáp án chuẩn) để tính toán các chỉ số: Exact Match (EM) và F1 Score.

Thực nghiệm Đối sánh (A/B Testing Baseline):

Chạy thử một model RAG cơ bản (chỉ dùng Vector Database, không dùng Graph) trên cùng tập dữ liệu.

So sánh điểm số F1 giữa RAG cơ bản và KG²RAG để chứng minh tính ưu việt của Đồ thị tri thức trong việc giải quyết hiện tượng "Hallucination" và các câu hỏi phức tạp (Multi-hop).

Phân tích Định tính (Qualitative Analysis):

Trích xuất 1-2 ví dụ cụ thể (Case Study) minh họa việc KG²RAG đã kết nối các văn bản như thế nào để tìm ra câu trả lời đúng, đưa vào báo cáo tổng kết.


Bạn cứ tiến hành commit code lên đi nhé. Xong xuôi, chúng ta sẽ bắt đầu xử lý tiếp khâu chạy Đánh giá cho các dataset còn lại và tính điểm F1!

Bạn viết 1 file readme.md khác trong folder KG2RAG với nội dung như này đi
