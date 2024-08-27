import google.generativeai as genai
import os
import json
import random
import time
from collections import Counter
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GEMINI_APIKEYS = os.getenv('GEMINI_APIKEYS').split(',')

datasets_dir = '../../../datasets'
save_json_path = './gemini_vqax.json'
checkpoint_dir = datasets_dir + '/VQA-X'

def get_most_common_answer(answers):
    return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]

def get_batch_prompt(sample_list):
    prompt_intro = (
        "You are a Vietnamese translator. Translate the following input strings into Vietnamese. "
        "The inputs include questions, answers, and explanations. Provide the translation directly without any additional commentary or analysis.\n"
        "Note: Return only the string of the translation for each input, separated by newlines. For example:\n"
        "Example:\n"
        "Input 1: Question: Is the window open?\nAnswer: no\nExplanation: the window shutters are closed\nExplanation: It's nighttime\n"
        "Input 2: Question: What color is the sky?\nAnswer: blue\nExplanation: it's a clear day\n"
        "Input 3: Question: How many dogs are there?\nAnswer: two\nExplanation: I can see two dogs in the park\nExplanation: They are playing fetch\n"
        "Output:\n"
        "Cửa sổ có mở không?\n"
        "không\n"
        "các cánh cửa sổ đóng\n"
        "Trời đã tối\n"
        "Bầu trời màu gì?\n"
        "xanh\n"
        "đó là một ngày quang đãng\n"
        "Có bao nhiêu con chó?\n"
        "hai\n"
        "Tôi có thể thấy hai con chó trong công viên\n"
        "Chúng đang chơi ném bắt\n"
        "These are the inputs:\n"
    )

    prompt_inputs = ""
    for i, sample in enumerate(sample_list, 1):
        prompt_inputs += f"Input {i}: Question: {sample['question']}\nAnswer: {sample['answer']}\n"
        for exp in sample['explanation']:
            prompt_inputs += f"Explanation: {exp}\n"
        prompt_inputs += "\n"

    return prompt_intro + prompt_inputs + "Output:\n"

def translate_batch(samples, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt_batch = get_batch_prompt(samples)
    try:
        response = model.generate_content(
            prompt_batch, 
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        translations = [t for t in response.text.strip().split('\n') if t]
        expected_translations = sum(len(sample['explanation']) for sample in samples) + 2 * len(samples)
        if len(translations) == expected_translations:
            return translations
        else:
            print(f"Unexpected number of translations: expected {expected_translations}, got {len(translations)}")
            return []
    except Exception as e:
        print(f"Error in translate_batch: {e}")
        return []

def save_processed_data(processed_data, dataset_name):
    output_file = f'vqaX_{dataset_name}_gemini.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Saved processed data to {output_file}")

def process_dataset(dataset_name, dataset_file):
    data_path = os.path.join(datasets_dir, 'VQA-X', dataset_file)
    
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    processed_data = {}
    error_list = []
    current_key_index = 0

    total_keys = len(data)
    
    for i in tqdm(range(0, total_keys, 4), desc=f"Processing {dataset_name} data", unit="batch"):
        if i % 400 == 0 and i != 0:
            save_processed_data(processed_data, dataset_name)
            time.sleep(random.randint(60,100))

        batch_samples = [
            {
                "question": data[key]["question"],
                "answer": get_most_common_answer(data[key]["answers"]),
                "explanation": data[key]["explanation"]
            }
            for key in list(data.keys())[i:i+4] if key in data
        ]

        api_key = GEMINI_APIKEYS[current_key_index % len(GEMINI_APIKEYS)]
        translations = translate_batch(batch_samples, api_key)
        current_key_index = (current_key_index + 1) % len(GEMINI_APIKEYS)
        
        if not translations:
            print(f"Failed to get translations for batch starting at index {i}. Skipping this batch.")
            error_list.extend([{'key': key, 'error': 'Translation failed'} for key in list(data.keys())[i:i+4] if key in data])
            continue
        
        translation_index = 0
        for j, sample in enumerate(batch_samples):
            key = list(data.keys())[i + j]
            item = data[key].copy()  # Create a copy of the original item
            
            # Add Vietnamese translations to the item
            item["question_vi_gemini"] = translations[translation_index]
            item["answer_vi_gemini"] = translations[translation_index + 1]
            item["explanation_vi_gemini"] = translations[translation_index + 2 : translation_index + 2 + len(item["explanation"])]
            
            translation_index += 2 + len(item["explanation"])
            
            try:
                processed_data[key] = item
            except Exception as e:
                print(f"An error occurred while processing key {key}: {e}")
                error_list.append({'key': key, 'error': str(e)})
        
        time.sleep(random.uniform(4, 10))
    
    # Final save of processed data
    save_processed_data(processed_data, dataset_name)
    
    # Save error list if there are any errors
    if error_list:
        error_file = f'vqaX_{dataset_name}_errors.json'
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_list, f, ensure_ascii=False, indent=2)
        print(f"Saved error list to {error_file}")

if __name__ == "__main__":
    dataset_files = {
        'test': 'vqaX_test.json',
        'train': 'vqaX_train.json',
        'val': 'vqaX_val.json'
    }

    for dataset_name, dataset_file in dataset_files.items():
        process_dataset(dataset_name, dataset_file)
    print("Processing complete.")