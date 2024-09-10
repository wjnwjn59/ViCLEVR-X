import os
import json
import time
import openai
from dotenv import load_dotenv
from collections import Counter
load_dotenv()
# Cấu hình
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

datasets_dir = '../../../datasets'
vqax_dir = os.path.join(datasets_dir, 'VQA-X')
def get_most_common_answer(answers):
    return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]
def prepare_batch_input_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as out_f:
        for key, value in data.items():
            common_answer = get_most_common_answer(value['answers'])
            messages = [
                {"role": "system", "content": "You are a Vietnamese translator. Translate each section separated by '|' to Vietnamese. Provide only the translations, each on a new line, without any additional text or explanations."},
                {"role": "user", "content": f"{value['question']}|{common_answer}|{'|'.join(value['explanation'])}"},
            ]
            request = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 1000
                }
            }
            out_f.write(json.dumps(request) + "\n")

def upload_file(file_path):
    try:
        with open(file_path, "rb") as f:
            batch_input_file = openai.files.create(
                file=f,
                purpose="batch"
            )
        return batch_input_file.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def create_batch(input_file_id):
    try:
        batch = openai.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Translation job"}
        )
        return batch.id
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None

def check_batch_status(batch_id):
    while True:
        batch_status = openai.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"Batch {batch_id} status: {status}", end="\r")
        
        if status in ["completed", "failed", "cancelled", "expired"]:
            return status
        time.sleep(60)

def download_batch_results(batch_id):
    try:
        batch_info = openai.batches.retrieve(batch_id)
        output_file_id = batch_info.output_file_id
        if not output_file_id:
            print("No output file found.")
            return None

        file_response = openai.files.content(output_file_id)
        
        output_file_path = f"{batch_id}_output.jsonl"
        with open(output_file_path, "w") as output_file:
            output_file.write(file_response.text)
        
        print(f"Results saved to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error downloading batch results: {e}")
        return None

def parse_batch_results(result_file, original_data):
    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]

    for result in results:
        key = result['custom_id']
        content = result['response']['body']['choices'][0]['message']['content']
        if '\n' in content:
            translations = content.split('\n')
            translations = [t for t in translations if t]
        elif '|' in content:
            translations = content.split('|')
            translations = [t for t in translations if t]
        
        if key in original_data:
            original_data[key]['question_vi_gpt'] = translations[0].strip()
            original_data[key]['answer_vi_gpt'] = translations[1].strip()
            original_data[key]['explanation_vi_gpt'] = [t.strip() for t in translations[2:]]

    return original_data

if __name__ == "__main__":
    dataset_files = {
        # 'test': 'vqaX_test.json',
        'train': 'vqaX_train.json',
        # 'val': 'vqaX_val.json'
    }
    for dataset_name, dataset_file in dataset_files.items():
        dataset_path = os.path.join(vqax_dir, dataset_file)
            
        prepare_batch_input_file(dataset_path, f'vqaX_{dataset_name}_batch.jsonl')
        
        # Upload và tạo batch
        batch_input_file_id = upload_file(f"vqaX_{dataset_name}_batch.jsonl")
        if not batch_input_file_id:
            print("Error uploading batch input file.")
            exit(1)
            
        batch_id = create_batch(batch_input_file_id)
        if not batch_id:
            print("Error creating batch.")
            exit(1)
            
        # Kiểm tra trạng thái và đợi hoàn thành
        final_status = check_batch_status(batch_id)
        if final_status != "completed":
            print(f"Batch processing failed with status: {final_status}")
            exit(1)
            
        # Tải về kết quả
        result_file = download_batch_results(batch_id)
        if not result_file:
            print("Error downloading batch results.")
            exit(1)
            
        # Đọc dữ liệu gốc
        with open(dataset_path, 'r') as f:
            original_data = json.load(f)
            
        # Parse kết quả và thêm vào dữ liệu gốc
        updated_data = parse_batch_results(result_file, original_data)
        
        # Lưu kết quả cuối cùng
        output_path = os.path.join(vqax_dir, f'vqaX_{dataset_name}_gpt.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
            
        print(f"Updated data saved to {output_path}")
        print("=====================================")
        