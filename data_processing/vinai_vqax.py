import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from collections import Counter
import random
from tqdm import tqdm
import re

torch.random.manual_seed(0)
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
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
tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
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
    # Chuẩn hóa các từ khóa
    text = re.sub(r'Câu trả lời:', 'Trả lời:', text, flags=re.IGNORECASE)
    text = re.sub(r'Lời giải thích:', 'Giải thích:', text, flags=re.IGNORECASE)
    
    # Tách văn bản bằng regex cải tiến
    parts = re.split(r'(Câu hỏi:|Trả lời:|Giải thích:)', text, flags=re.IGNORECASE)
    
    result = {"question": "", "answer": "", "explanation": []}
    current_key = ""
    
    for part in parts:
        part = part.strip()
        lower_part = part.lower()
        
        if lower_part in ["câu hỏi:", "trả lời:", "giải thích:"]:
            if lower_part == "câu hỏi:":
                current_key = "question"
            elif lower_part == "trả lời:":
                current_key = "answer"
            elif lower_part == "giải thích:":
                current_key = "explanation"
        elif current_key:
            if current_key == "explanation":
                # Tách nhiều câu giải thích nếu có
                explanations = re.split(r'(?<=[.!?])\s+', part)
                result[current_key].extend([exp.strip() for exp in explanations if exp.strip()])
            else:
                result[current_key] = part.strip()
    
    # Xử lý trường hợp giải thích bị kết hợp với câu trả lời
    if not result["explanation"] and ". " in result["answer"]:
        answer_parts = result["answer"].split(". ", 1)
        if len(answer_parts) == 2:
            result["answer"] = answer_parts[0].strip()
            result["explanation"] = [answer_parts[1].strip()]
    
    return result

def translate_dataset(dataset):
    translated_dataset = {}
    batch_size = 30

    dataset_items = list(dataset.items()) if isinstance(dataset, dict) else enumerate(dataset)

    for i in tqdm(range(0, len(dataset_items), batch_size)):
        batch = dataset_items[i:i+batch_size]
        
        # Prepare texts for translation (questions and answers)
        qa_texts = []
        explanation_texts = []
        for _, item in batch:
            question = item['question']
            answer_counts = Counter(ans['answer'] for ans in item['answers'])
            most_common_answer = answer_counts.most_common(1)[0][0]
            qa_text = f"Question: {question} Answer: {most_common_answer}"
            qa_texts.append(qa_text)
            
            # Prepare explanations for separate translation
            for explanation in item['explanation']:
                explanation_texts.append(explanation)
        
        # Translate questions and answers
        translated_qa_texts = translate_en2vi(qa_texts)
        
        # Translate explanations
        translated_explanation_texts = translate_en2vi(explanation_texts)
        
        explanation_index = 0
        for j, (key, item) in enumerate(batch):
            translated_item = item.copy()
            
            # Use the improved split_translated_text function
            split_result = split_translated_text(translated_qa_texts[j])
            translated_item['question_vi_vinai'] = split_result['question']
            translated_item['answer_vi_vinai'] = split_result['answer']
            
            # Add translated explanations
            translated_item['explanation_vi_vinai'] = []
            for _ in range(len(item['explanation'])):
                if explanation_index < len(translated_explanation_texts):
                    translated_item['explanation_vi_vinai'].append(translated_explanation_texts[explanation_index])
                    explanation_index += 1
            
            translated_dataset[key] = translated_item
    
    return translated_dataset

# Translate datasets
translated_train = translate_dataset(train_data)
translated_val = translate_dataset(val_data)
translated_test = translate_dataset(test_data)

# Save translated datasets
def save_translated_data(data, file_name):
    output_dir = vqax_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_translated_data(translated_train, 'vqaX_train_vinai.json')
save_translated_data(translated_val, 'vqaX_val_vinai.json')
save_translated_data(translated_test, 'vqaX_test_vinai.json')

print("Translation completed and saved.")