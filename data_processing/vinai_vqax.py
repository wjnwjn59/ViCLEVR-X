import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from collections import Counter
import random
from tqdm import tqdm
import re

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
os.environ["WORLD_SIZE"] = '2'

# Load datasets
datasets_dir = '../../../datasets'
vqax_dir = os.path.join(datasets_dir, 'VQA-X')
train_dir = os.path.join(vqax_dir, 'vqaX_train.json')
test_dir = os.path.join(vqax_dir, 'vqaX_test.json')
val_dir = os.path.join(vqax_dir, 'vqaX_val.json')

def load_data(file_path):
    with open(file_path) as f:
        return json.load(f)

train_data = load_data(train_dir)
test_data = load_data(test_dir)
val_data = load_data(val_dir)

# Initialize translator
tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX", tgt_lang="vi_VN")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)

def translate_en2vi(en_texts):
    input_ids = tokenizer_en2vi(en_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return vi_texts

def split_translated_text(text):
    parts = re.split(r'(Câu hỏi:|Trả lời:|Giải thích:)', text)
    result = {"question": "", "answer": "", "explanation": []}
    current_key = ""
    
    for part in parts:
        part = part.strip()
        if part.lower() in ["câu hỏi:", "trả lời:", "câu trả lời:", "giải thích:", "lời giải thích:"]:
            if part.lower() in ["trả lời:", "câu trả lời:"]:
                current_key = "answer"
            elif part.lower() in ["giải thích:", "lời giải thích:"]:
                current_key = "explanation"
            else:
                current_key = "question"
        elif current_key:
            if current_key == "explanation":
                result[current_key].append(part)
            else:
                result[current_key] = part
    
    return result

def translate_dataset(dataset):
    translated_dataset = {}
    batch_size = 30

    # Convert dataset to list of tuples (key, value) if it's a dict
    dataset_items = list(dataset.items()) if isinstance(dataset, dict) else enumerate(dataset)

    for i in tqdm(range(0, len(dataset_items), batch_size)):
        batch = dataset_items[i:i+batch_size]
        
        # Prepare texts for translation
        combined_texts = []
        for _, item in batch:
            question = item['question']
            answer_counts = Counter(ans['answer'] for ans in item['answers'])
            most_common_answer = answer_counts.most_common(1)[0][0]
            explanations = ' Giải thích: '.join(item['explanation'])
            combined_text = f"Câu hỏi: {question} Trả lời: {most_common_answer} Giải thích: {explanations}"
            combined_texts.append(combined_text)
        
        # Translate combined texts
        translated_combined_texts = translate_en2vi(combined_texts)
        
        for j, (key, item) in enumerate(batch):
            translated_item = item.copy()
            
            # Split the translated text
            split_result = split_translated_text(translated_combined_texts[j])
            translated_item['question_vi_vinai'] = split_result['question']
            translated_item['answer_vi_vinai'] = split_result['answer']
            translated_item['explanation_vi_vinai'] = split_result['explanation']
            
            translated_dataset[key] = translated_item
    
    return translated_dataset

# Translate datasets
translated_train = translate_dataset(train_data)
translated_val = translate_dataset(val_data)
translated_test = translate_dataset(test_data)

# Save translated datasets
def save_translated_data(data, file_name):
    output_dir = os.path.join(vqax_dir, 'vinai_2')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_translated_data(translated_train, 'vqaX_train_vinai.json')
save_translated_data(translated_val, 'vqaX_val_vinai.json')
save_translated_data(translated_test, 'vqaX_test_vinai.json')

print("Translation completed and saved.")