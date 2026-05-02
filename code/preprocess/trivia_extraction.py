import os
import re  # Thêm thư viện xử lý chuỗi để sửa lỗi tên file
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    
    resp = llm.complete(query)
    resp = resp.text
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
data_path = '../../data/trivia_qa/trivia.json'
with open(data_path) as f:
    data = json.load(f)

triplets = {}

# 1. Đổi model sang bản 1b, giới hạn VRAM (num_ctx: 4096) và tăng timeout lên 600s
llm = Ollama(model='llama3.2:1b', request_timeout=600.0, additional_kwargs={"num_ctx": 4096})
print(f"====== ĐANG GỌI MODEL: {llm.model} ======")

out_dir = '../../data/trivia_qa/kgs/extracted_subkgs'
os.makedirs(out_dir, exist_ok=True)

# 2. Khởi tạo tính năng Resume: Quét và báo cáo file đã hoàn thành
existing_files = [f for f in os.listdir(out_dir) if f.endswith('.json')]
print(f"[*] Đã tìm thấy {len(existing_files)} thực thể được xử lý từ trước.")
print("[*] Hệ thống sẽ tua nhanh qua các dữ liệu này...\n")

count = 0

for sample in tqdm(data, desc="Tiến trình TriviaQA"):
    question = sample['question']
    answer = sample['answer']
    ctxs = sample['context']
    cands = [sp[0] for sp in sample['supporting_facts']]
    
    for ctx in ctxs:
        ent = ctx[0]
        
        if (ent in triplets):
            continue
        if not ent in cands:
            continue
            
        # 3. Fix lỗi tên file chứa ký tự đặc biệt của Windows
        safe_ent = re.sub(r'[<>:"/\\|?*]', '_', ent)
        out_path = os.path.join(out_dir, f'{safe_ent}.json')
        
        # 4. Tính năng Resume chống file rỗng (0 KB)
        if os.path.exists(out_path):
            if os.path.getsize(out_path) > 0:
                continue
            else:
                os.remove(out_path) # Xóa file rỗng do bị ngắt đột ngột lần trước
                
        triplets[ent] = {}
        
        for i in range(len(ctx[1])):
            ctx_text = f'{ent}: {ctx[1][i]}'
            try:
                ext_triplets = extract_triplets(llm, ctx_text)
            except Exception as e:
                print(f'\n[Cảnh báo] Lỗi Timeout. Bỏ qua câu. Chi tiết: {e}')
                continue
                
            if len(ext_triplets)==0:
                continue
            triplets[ent][i] = ext_triplets
            
        with open(out_path,'w') as f:
            json.dump(triplets[ent], f)
            count += 1

print(f'\nNewly extract entity KGs number: {count}')