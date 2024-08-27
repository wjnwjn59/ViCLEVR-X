from googletrans import Translator
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import logging
import time
import random
# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set paths
datasets_dir = '../../../datasets'
clevr_dir = os.path.join(datasets_dir,'CLEVR/CLEVR_v1.0')
clevrx_dir = os.path.join(datasets_dir,'CLEVR-X')

questions_dir = os.path.join(clevr_dir, 'questions')
image_dir = os.path.join(clevr_dir, 'images')

train_image_dir = os.path.join(image_dir, 'train')
val_image_dir = os.path.join(image_dir, 'val')



# Read data
with open(os.path.join(clevrx_dir, 'CLEVR_train_explanations_v0.7.10.json')) as f:
    explain_train = json.load(f)['questions']
with open(os.path.join(clevrx_dir, 'CLEVR-X_train_translated_temp_60_64.json')) as f:
    already_translated = json.load(f)['questions']

# with open(os.path.join(clevrx_dir, 'CLEVR_val_explanations_v0.7.10.json')) as f:
#     explain_val = json.load(f)
print('Data loaded')

# Tạo set các question_index đã được dịch
translated_indices = set(item['question_index'] for item in already_translated)
print(f"Already translated: {len(translated_indices)} items")
error_items = []

def translate_chunk(chunk, chunk_index):
    translator = Translator()
    translated_chunk = []
    for i, data in enumerate(chunk):
        if data['question_index'] in translated_indices:
            translated_chunk.append(data)
            continue
        try:
            translated_data = data.copy()
            translated_data['question_vi_ggtrans'] = translator.translate(data['question'], src='en', dest='vi').text
            translated_data['answer_vi_ggtrans'] = translator.translate(data['answer'], src='en', dest='vi').text
            translated_data['factual_explanation_vi_ggtrans'] = [translator.translate(explanation, src='en', dest='vi').text for explanation in data['factual_explanation']]
            translated_chunk.append(translated_data)
            # time.sleep(random.uniform(2, 4))
            sys.stdout.write(f"\rChunk {chunk_index}: {i+1}/{len(chunk)} ({(i+1)/len(chunk)*100:.2f}%)")
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"Error at chunk {chunk_index}, item {i}: {e}")
            error_items.append((chunk_index, i, data))
    return translated_chunk

def translate_explain(explain_data, timeout=0.5, save_interval=1):
    num_cores = 40
    chunk_size = len(explain_data) // num_cores + (1 if len(explain_data) % num_cores else 0)
    chunks = [explain_data[i:i + chunk_size] for i in range(0, len(explain_data), chunk_size)]

    translated_data = already_translated  # Bắt đầu với dữ liệu đã dịch
    completed_chunks = set()
    
    def save_progress(data, filename):
        with open(os.path.join(clevrx_dir, filename), 'w') as f:
            json.dump({'questions': data}, f)
        logging.info(f"Progress saved to {filename}")

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_to_chunk = {executor.submit(translate_chunk, chunk, idx): idx for idx, chunk in enumerate(chunks)}

        completed_futures = 0
        total_futures = len(future_to_chunk)
        start_time = time.time()
        last_save_time = start_time

        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result(timeout=timeout*3600)  # timeout in hours
                translated_data.extend([item for item in result if item['question_index'] not in translated_indices])
                completed_chunks.add(chunk_idx)
                completed_futures += 1
                sys.stdout.write(f"\rTranslating: {completed_futures}/{total_futures} chunks completed")
                sys.stdout.flush()
                
                # Lưu tiến độ định kỳ
                current_time = time.time()
                if current_time - last_save_time > save_interval * 3600 or completed_futures >= 32:
                    temp_filename = f'CLEVR-X_train_translated_temp_{completed_futures}_{total_futures}.json'
                    save_progress(translated_data, temp_filename)
                    last_save_time = current_time

            except TimeoutError:
                logging.error(f"Timeout for chunk {chunk_idx}")
            except Exception as e:
                logging.error(f"Error completing future for chunk {chunk_idx}: {e}")

    return translated_data

print('Translating train data...')
explain_train_translated = translate_explain(explain_train)

# Save final data
final_filename = os.path.join(clevrx_dir, 'CLEVR-X_train_translated_final.json')
with open(final_filename, 'w') as f:
    json.dump({'questions': explain_train_translated}, f)
print(f"\nTranslation completed. Final data saved to {final_filename}")

# Print statistics
print(f"Total items processed: {len(explain_train_translated)}")
print(f"Total error items: {len(error_items)}")

with open(os.path.join(clevrx_dir, 'error_items.json'), 'w') as f:
    json.dump({'error_items': error_items}, f)
print('Error items saved')