import os
import sys
import string
import argparse
import pandas as pd
import ujson as json
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

def ngram_overlap(span,sent,n=3):
    while (len(span)<n) or (len(sent)<n):
        n -= 1
    if n<=0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i+n] for i in range(len(span_tokens)-n+1)])
    sent_tokens = set([sent_tokens[i:i+n] for i in range(len(sent_tokens)-n+1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap)+0.01)/(len(span_tokens)+0.01))

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    try:
        resp = llm.complete(query)
        resp = resp.text
    except Exception as e:
        print(f"\n[Cảnh báo] Lỗi gọi AI, bỏ qua câu. Chi tiết: {e}")
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


import traceback # Thêm thư viện này ở đầu file nếu chưa có để in lỗi chi tiết

def extract_triplets_from_musique(data, llm, kg_path):
    ents = set()
    ent2text = dict()
    text2seq = dict()
    
    # --- TÍNH NĂNG RESUME (KHÔI PHỤC) ---
    ent2triplets = dict()
    if os.path.exists(kg_path):
        try:
            with open(kg_path, 'r') as f:
                content = f.read()
                if content.strip(): # Kiểm tra file không rỗng
                    ent2triplets = json.loads(content)
                    print(f"[*] Đã khôi phục thành công {len(ent2triplets)} thực thể từ lần chạy trước.")
                    print("[*] Đang tua nhanh qua các dữ liệu đã hoàn thành...\n")
        except Exception as e:
            print(f"[!] Lỗi khi đọc file cũ: {e}. Sẽ tạo file mới.")

    textcount = 0
    unique_textcount = 0
    
    # Sử dụng iterrows() có progress bar
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Tiến trình MuSiQue"):
        question = row['question']
        answer = row['answer']
        ctxs = row['paragraphs']
        current_ents = [ctx['title'] for ctx in ctxs]
        current_ents = sorted(current_ents, key=lambda x: len(x), reverse=True)

        for ctx in ctxs:
            ent = ctx['title']
            text = ctx['paragraph_text']
            ents.add(ent)
            
            if ent not in ent2text:
                ent2text[ent] = set()
            if text not in ent2text[ent]:
                ent2text[ent].add(text)
                unique_textcount += 1
            if ent not in text2seq:
                text2seq[ent] = dict()
            if text not in text2seq[ent]:
                text2seq[ent][text] = len(text2seq[ent])
            
            ctx['seq'] = text2seq[ent][text]
            textcount += 1

            # --- KIỂM TRA RESUME CHO TỪNG ĐOẠN VĂN ---
            # Nếu thực thể (ent) và đoạn văn (seq) đã có trong dict -> bỏ qua không gọi LLM nữa
            seq_str = str(ctx['seq']) # Ép kiểu chuỗi vì JSON key luôn là chuỗi
            if ent in ent2triplets and seq_str in ent2triplets[ent]:
                continue 

            # Gọi LLM trích xuất (Đã được bọc try...except ở hàm extract_triplets như bạn đã sửa)
            triplets = extract_triplets(llm, text)
            
            if len(triplets) <= 0:
                continue
                
            if ent not in ent2triplets:
                ent2triplets[ent] = dict()
            if seq_str not in ent2triplets[ent]:
                ent2triplets[ent][seq_str] = list()
                
            # Gộp 2 list lại với nhau
            combined_list = ent2triplets[ent][seq_str] + triplets
            
            # Loại bỏ các List con trùng lặp bằng cách chuyển tạm thành Tuple rồi ép ngược lại List
            unique_triplets = [list(t) for t in set(tuple(element) for element in combined_list)]
            
            ent2triplets[ent][seq_str] = unique_triplets
            
            # --- LƯU LIÊN TỤC (INCREMENTAL SAVING) ---
            # Ghi đè file JSON ngay lập tức sau mỗi lần cập nhật. Dù cúp điện cũng không mất.
            with open(kg_path, 'w') as f:
                json.dump(ent2triplets, f)

    print(f'\n#ents: {len(ents)}')
    print(f'#total text: {textcount}')
    print(f'#unique text: {unique_textcount}')
    return data, ent2triplets


def main(args):
    # Dùng bản 1b cho nhanh và vừa RAM, tăng timeout
    model_name = 'llama3.2:1b'
    token_counter = TokenCountingHandler()
    Settings.llm = Ollama(model=model_name, request_timeout=600.0, additional_kwargs={"num_ctx": 4096})
    Settings.callback_manager = CallbackManager([token_counter])

    data_dir = '../../data/MuSiQue'
    data_path = os.path.join(data_dir, 'musique_ans_v1.0_dev_mapped.jsonl')
    
    if not os.path.exists(data_path):
        print(f'Data file not found: {data_path}')
        return
        
    data = pd.read_json(data_path, lines=True)

    out_dir = '../../data/MuSiQue/kgs/extract_subkgs'
    os.makedirs(out_dir, exist_ok=True)
    
    kg_path = os.path.join(out_dir, 'musique_kg.json')
    
    # Truyền thêm kg_path vào hàm
    mapped_data, ent2triplets = extract_triplets_from_musique(data, Settings.llm, kg_path)

    print(f'\nCompletion token count: {token_counter.completion_llm_token_count}')
    print(f'Prompt token count: {token_counter.prompt_llm_token_count}')

    # Cập nhật file mapped_data (File này chỉ lưu thông tin cấu trúc, lưu 1 lần cuối là được)
    mapped_data_path = os.path.join(out_dir, 'musique_ans_v1.0_dev_mapped.jsonl')
    print(f'Saving mapped data to {mapped_data_path}')
    mapped_data.to_json(mapped_data_path, orient='records', lines=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)