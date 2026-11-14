# Báo cáo cá nhân — Day 6

> **MSSV / Họ tên:** 2A202603927 / Đàm Hồng Thái

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** Nội dung đại số, bài tập và ví dụ trong *A Review of Algebra* của Romeyn Henry Rivenburg (1914), [Project Gutenberg eBook #38536](https://www.gutenberg.org/ebooks/38536). Manifest ghi sách thuộc public domain tại Hoa Kỳ theo giấy phép Project Gutenberg; ngoài Hoa Kỳ cần kiểm tra luật địa phương. Mục lục ghi phần sách ở trang in 7–80 (74 trang danh nghĩa), nhưng nguồn UTF-8 không có phân trang ổn định. Corpus lấy dòng 202–3602, loại phần đầu/cuối không thuộc nội dung, chia theo đoạn văn liên tiếp thành **54 unit** truy vết tương đương trang; mỗi unit có ít nhất 1.500 ký tự. Tổng văn bản chuẩn hóa là **85.415 ký tự**. Vì vậy 74 là số trang in theo mục lục, còn 54 là số unit dùng trong pipeline, không phải số trang PDF. Xem [manifest](../../starter/data/gutenberg_38536_manifest.json) và [corpus JSONL](../../starter/data/gutenberg_38536_selected.jsonl).
- **Chuẩn hóa, lỗi và giới hạn:** Đọc UTF-8, loại BOM, chuẩn hóa Unicode NFC; giữ ID unit và dòng nguồn. Đo ký tự bằng số code point Python trong trường text JSONL sau chuẩn hóa, gồm dấu cách và xuống dòng nội bộ. Không dùng OCR. `ingestion_errors.jsonl` hiện rỗng. Biểu thức toán lấy từ bản plain-text, nên ký pháp phức tạp cần đối chiếu bản HTML.
- **Embedding, vector store, lệnh chạy và chi phí:** OpenRouter `openai/text-embedding-3-small`, 1536 chiều; cùng cách xử lý văn bản cho query/document. Không có revision bất biến của model trong artifacts. PostgreSQL **16.16** + pgvector **0.8.7**, bảng `rag_chunks`, cosine distance, exact sequential scan, không dùng ANN index. Môi trường Python 3.11.5; psycopg 3.3.8, pypdf 6.21.4, python-dotenv 1.2.4. Lệnh chạy từ thư mục `starter/`: `python -m src.run`. Embedding ghi nhận 71.326 token (70.257 token tài liệu A+B và 1.069 token query dùng chung); ước tính **$0.00142652** theo $0.02/triệu token. Đây là ước tính, hóa đơn thực tế có thể khác. C tái sử dụng vector tài liệu của B nên không tạo thêm embedding tài liệu.
- **Metadata bắt buộc và trường lọc:** `document_id` (ID manifest), `source` (URL nguồn), `page` (unit ID ổn định), `source_start_line` và `source_end_line` (dòng nguồn), `section` (heading nguồn viết hoa/đánh dấu hoặc `null`). `section` được lấy từ heading gần nhất trước đó theo thứ tự nguồn, kể cả qua ranh giới unit; dùng exact-match filter khi câu hỏi cung cấp section. Các trường truy vết lấy từ manifest/JSONL; section lấy từ heading trong nguồn, không suy đoán. Metadata hỗ trợ truy nguyên và lọc, nhưng section kế thừa từ heading có thể thiếu hoặc không hoàn toàn hữu ích về mặt ngữ nghĩa.

## Cấu hình và benchmark

- **A:** cửa sổ cố định 500 ký tự, overlap 50.
- **B:** nhóm theo section/đoạn văn, tối đa 900 ký tự.
- **C:** cùng chunk và vector như B; chỉ thêm exact filter trên `section` khi câu hỏi nêu filter. Đây là khác biệt duy nhất giữa B và C.
- **Bộ câu hỏi và evidence:** [24 câu](../../starter/data/benchmark_questions.json), chia cố định **18 dev / 6 test**; có 8 câu cung cấp filter (4 dev, 4 test), trong đó T02, T03, T04 thuộc mẫu test T01–T05 được chấm thủ công. Các câu hỏi được tạo trước các lần truy xuất; evidence được kiểm tra là substring chính xác trong unit nguồn đã chuẩn hóa. Filter do câu hỏi cung cấp trước khi tìm kiếm và được xác thực với heading có trong corpus, không lấy từ gold location. Hit yêu cầu chunk ở đúng source/page và chứa đủ quote sau chuẩn hóa khoảng trắng; không ghép evidence bị tách qua nhiều chunk. [Trace top-3](../../starter/results/traces.jsonl) lưu câu hỏi, filter, hits, metadata và nhãn match.
- **Tiêu chí chọn chiến lược:** Log ghi quy tắc dev-first: ưu tiên Evidence Hit@3 trên dev, sau đó grounding mẫu, rồi tài nguyên thấp hơn. Tuy nhiên quy tắc chọn được ghi nhận sau khi điểm test đã hiển thị; vì vậy đây là tiêu chí hậu nghiệm, không phải preregistration mù trước test. Kết quả test chỉ nên xem là mô tả.

## Kết quả

### Truy xuất và tài nguyên

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian ingest | Chi phí embedding tài liệu* |
|---|---|---:|---:|---:|---:|---:|
| A | dev | 4/18 (0.222) | 7/18 (0.389) | 219 | 22.77 s | $0.00073512 |
| A | test | 3/6 (0.500) | 5/6 (0.833) | 219 | 22.77 s | $0.00073512 |
| B | dev | 4/18 (0.222) | 10/18 (0.556) | 168 | 21.87 s | $0.00067002 |
| B | test | 3/6 (0.500) | 4/6 (0.667) | 168 | 21.87 s | $0.00067002 |
| C | dev | 4/18 (0.222) | 10/18 (0.556) | 168 | 0.43 s | $0.00000000 |
| C | test | 3/6 (0.500) | 3/6 (0.500) | 168 | 0.43 s | $0.00000000 |

*\* Chi phí cột này chỉ tính embedding tài liệu theo mức giá giả định $0.02/triệu token. Query embedding 1.069 token ($0.00002138) được tính một lần và dùng chung; tổng chi phí ước tính toàn lần chạy là $0.00142652. Thời gian C đo lần ingest tái sử dụng vector B, không phải một lần tạo vector độc lập.*

Độ dài chunk: A min/median/max **77/500/500**, tổng 93.467 ký tự; B và C **15/595/899**, tổng 85.187 ký tự. Nạp lặp lại giữ nguyên số hàng (A 219, B/C 168), kiểm tra idempotency đạt cho cả ba.

### Đánh giá mẫu thủ công

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N) | Grounding 0/1/2 (số câu) | Grounding trung bình /2 |
|---|---:|---:|---:|---:|---:|
| A | 1.20/2 (10) | 15/15 | 1/3 | 1/0/4 | 1.60/2 |
| B | 1.50/2 (10) | 15/15 | 0/3 | 1/1/3 | 1.40/2 |
| C | 1.50/2 (10) | 15/15 | 3/3 | 1/1/3 | 1.40/2 |

Coherence lấy chunk tại offset 0 của 10 unit đầu theo thứ tự ID; grounding và metadata dùng top-3 của năm câu test đầu (T01–T05), tức 15 hit mỗi chiến lược. Filter compliance chỉ tính ba câu trong mẫu có filter. Metadata validity đối chiếu `source`, `page`, dòng nguồn và `section` với chunk/corpus, không phải kiểm toán độc lập của người chấm. Nhãn coherence/grounding là AI-assisted, không blind và không phải đánh giá độc lập của tác giả; cần rà soát trước khi coi là nhãn thủ công cuối cùng. Chi tiết và lý do từng mẫu: [manual_coherence.csv](../../starter/results/manual_coherence.csv), [manual_grounding.csv](../../starter/results/manual_grounding.csv).

### Trace và nhãn

- Trace đủ top-3: [starter/results/traces.jsonl](../../starter/results/traces.jsonl); số liệu và cấu hình: [starter/results/metrics.json](../../starter/results/metrics.json); lỗi ingestion: [starter/results/ingestion_errors.jsonl](../../starter/results/ingestion_errors.jsonl).
- Bộ mẫu coherence là 10 unit nguồn đầu theo ID, cùng offset 0 cho A/B/C. Mẫu grounding/metadata là T01–T05 và toàn bộ top-3. Nhãn được tạo bằng AI assistance, kiểm tra theo nguồn/evidence; kết quả truy xuất đã hiển thị trước khi chấm, nên có nguy cơ thiên lệch.

## Phân tích và kết luận

- **A–B:** Hit@1 không đổi. B tăng dev Evidence Hit@3 từ 7/18 lên 10/18 và coherence mẫu từ 1.20 lên 1.50; đổi lại test Evidence Hit@3 giảm 5/6 xuống 4/6, grounding trung bình giảm 1.60 xuống 1.40. B tạo ít chunk hơn (168 so với 219), ít token embedding tài liệu hơn (33.501 so với 36.756) và thời gian ingest lần này ngắn hơn (21.87 so với 22.77 giây).
- **B–C:** Chunk, vector, coherence và grounding không đổi. Dev Evidence Hit@3 giữ 10/18; test giảm 4/6 xuống 3/6. Trong ba câu test có filter thuộc mẫu thủ công, tuân thủ tăng 0/3 lên 3/3. C tái sử dụng vector của B, nên thời gian 0.43 giây và không phát sinh token embedding tài liệu mới.
- **Hai ví dụ lỗi:**  
  1. **T01** — cả A/B/C đều không truy xuất đủ evidence về thứ tự phép toán; gold nằm ở `gutenberg-38536-u005`, dòng 489–568, gồm “Next, multiplication and division” và “Last of all, addition and subtraction”.  
  2. **T02** — với B/C, top-3 có ví dụ/kết luận nhưng thiếu quy tắc tổng quát và điều kiện đầy đủ “If b^2 - 4ac is negative, the roots are imaginary and unequal.”; vì vậy không đạt exact Evidence Hit. Gold ở `gutenberg-38536-u024`, dòng 1790–1850. A tìm được bằng chứng đầy đủ trong mẫu. Có thể tra ID câu và từng hit trong trace.
- **Lựa chọn và đánh đổi:** Theo tiêu chí dev-first đã ghi (nhưng ghi hậu nghiệm), chọn **C** cho trường hợp cần truy vấn có filter section: nó ngang B về dev Evidence Hit@3/grounding mẫu và đạt 3/3 filter compliance trong mẫu mà không tạo lại embedding tài liệu. Nếu mục tiêu chỉ là test Evidence Hit@3 của bộ nhỏ này, A đạt 5/6 và B đạt 4/6, cao hơn C 3/6. Không có chiến lược thắng mọi tiêu chí; chọn C chỉ phù hợp khi tuân thủ filter là yêu cầu.
- **Hạn chế:** Chỉ một cuốn sách và sáu câu test; outline đầu sách có thể rò rỉ khái niệm của chương sau, mỗi câu làm tỷ lệ test thay đổi đáng kể. Evidence Hit dùng exact quote nên có thể bỏ sót bằng chứng tương đương hoặc evidence trải qua nhiều chunk. Nhãn thủ công nhỏ, AI-assisted, không mù và metadata validity chủ yếu là đối chiếu theo chính dữ liệu/chunk. Thời gian phụ thuộc lần chạy/mạng; thời gian C không so sánh ngang với lần tạo embedding mới. Giá API là ước tính, không phải hóa đơn.
- **Nếu mở rộng:** Chốt chiến lược và tiêu chí trước khi mở test; bổ sung test theo nhiều sách/nguồn và nhiều section, tăng số câu có filter, kiểm tra nhãn bởi người chấm độc lập và lưu kết quả phân xử. Rà soát heading tự động, kiểm tra ký pháp đại số với bản HTML, đồng thời đo riêng thời gian chunking, embedding, lưu và truy vấn qua nhiều lần chạy.

