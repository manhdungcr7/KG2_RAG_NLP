import os
import re
import ast
import ujson as json
import pandas as pd
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    
    try:
        resp = llm.complete(query)
        resp = resp.text
    except Exception as e:
        print(f"\n[Cảnh báo] Lỗi Timeout. Bỏ qua câu. Chi tiết: {e}")
        return []

    triplets = set()
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

# ---------------------------------------------------------
# XỬ LÝ CHÍNH
# ---------------------------------------------------------
# Đường dẫn tới file dev.parquet bạn vừa tải về
data_path = '../../data/2WikiMultihopQA/dev.parquet'
data = pd.read_parquet(data_path)

triplets = {}

# Cấu hình Model nhỏ + Giới hạn 4GB VRAM + Tăng Timeout chống sập
llm = Ollama(model='llama3.2:1b', request_timeout=600.0, additional_kwargs={"num_ctx": 4096})
print(f"====== ĐANG GỌI MODEL: {llm.model} ======")

out_dir = '../../data/2WikiMultihopQA/kgs/extracted_subkgs'
os.makedirs(out_dir, exist_ok=True)

# Khởi tạo tính năng Resume
existing_files = [f for f in os.listdir(out_dir) if f.endswith('.json')]
print(f"[*] Đã tìm thấy {len(existing_files)} thực thể được xử lý từ trước.")
print("[*] Hệ thống sẽ tua nhanh qua các dữ liệu này...\n")

count = 0

# Duyệt toàn bộ file Parquet bằng iterrows
for index, sample in tqdm(data.iterrows(), total=len(data), desc="Tiến trình 2Wiki"):
    
    # 1. Dùng ast để "ép" chuỗi văn bản ngụy trang trở lại thành mảng (List) xịn
    try:
        ctxs = ast.literal_eval(sample['context'])
    except:
        ctxs = sample['context'] # Đề phòng máy bạn đã tự parse
        
    for ctx in ctxs:
        ent = ctx[0]
        paragraph_text = ctx[1] # Ở 2Wiki, đây là nguyên 1 đoạn văn dài, không bị cắt nhỏ
        
        if ent in triplets:
            continue
            
        safe_ent = re.sub(r'[<>:"/\\|?*]', '_', ent)
        out_path = os.path.join(out_dir, f'{safe_ent}.json')
        
        # Tính năng Resume
        if os.path.exists(out_path):
            if os.path.getsize(out_path) > 0:
                continue
            else:
                os.remove(out_path)
                
        triplets[ent] = {}
        
        # 2. Xóa bỏ vòng lặp for i in range(...) cũ
        # Gom toàn bộ đoạn văn ném cho LLM xử lý 1 lần duy nhất cho nhanh
        ctx_text = f'{ent}: {paragraph_text}'
        
        ext_triplets = extract_triplets(llm, ctx_text)
        
        if len(ext_triplets) > 0:
            triplets[ent][0] = ext_triplets # Lưu thẳng vào vị trí [0]
            
        with open(out_path,'w') as f:
            json.dump(triplets[ent], f)
            count += 1

print(f'\nNewly extract entity KGs number: {count}')