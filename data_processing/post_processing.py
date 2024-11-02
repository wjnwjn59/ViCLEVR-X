import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from collections import Counter

# Define paths
dataset_dir = "../../../datasets/VQA-X"
sampling_dir = os.path.join(dataset_dir, "study_case_add_llm/sampling_20")
output_dir = os.path.join(dataset_dir, "final_data/post_processing")
os.makedirs(output_dir, exist_ok=True)
ner_ids_output_dir = os.path.join(output_dir, "ner_answers.json")
item_per_batch = 5

# Set up CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WORLD_SIZE"] = "1"

# Load model and tokenizer for NER
model_name = "dslim/bert-base-NER-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize NER pipeline with GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0, batch_size=item_per_batch)

# Function to get the most common answer
def get_most_common_answer(answers):
    return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]

# Function to check for named entities in the answer part only
# Now returns additional information (combined_text, entity, answer)
def contains_named_entity_in_answer(question, answer):
    answer_special =["zoo", "polar", "crow", "kite", "rock", "navy", "carrot", "tripod", "army", "steam", "teddy"]
    if answer in answer_special:
        return []
    combined_text = f"{question} {answer}"
    ner_results = ner_pipeline(combined_text)

    # Calculate answer position in combined_text
    answer_start = combined_text.find(answer)
    answer_end = answer_start + len(answer)
    
    entities = ["B-PER", "I-PER", "B-ORG", "I-ORG"]
    found_entities = []
    
    # Check if any entity is within the answer bounds and collect details
    for ent in ner_results:
        if ent["start"] >= answer_start and ent["end"] <= answer_end and ent["entity"] in entities:
            found_entities.append(combined_text)
    
    return found_entities

# Main processing loop
ner_answers = []
for dataset in ["train", "val", "test"]:
    # Load original and sampling data
    with open(os.path.join(dataset_dir, f"vqaX_{dataset}.json"), "r") as f:
        original_data = json.load(f)
    sampling_data = pd.read_csv(os.path.join(sampling_dir, f"{dataset}.csv"))

    for index in tqdm(range(len(sampling_data)), desc=f"Processing {dataset}"):
        sample = sampling_data.iloc[index]
        question_id = str(sample["question_id"])
        original_question = original_data[question_id]["question"]
        original_answer = get_most_common_answer(original_data[question_id]["answers"])
        img_id = original_data[question_id]["image_id"]

        # Check if there is a named entity specifically in the answer part
        ner_entities = contains_named_entity_in_answer(original_question, original_answer)
        if ner_entities:
            sampling_data.at[index, "answer"] = original_answer
            ner_answers.extend(ner_entities)  # Save the details of NER entities

        # Add image ID to the sampling data
        sampling_data.at[index, "img_id"] = img_id

    # remove columns
    sampling_data = sampling_data.drop(columns=['question_selection', 'answer_selection', 'explanation_selection'])
    sampling_data.to_csv(os.path.join(output_dir, f"vqaX_{dataset}_translated.csv"), index=False)

# Save the list of NER answers with combined text, entity type, and answer
with open(ner_ids_output_dir, "w") as ner_file:
    json.dump(ner_answers, ner_file, ensure_ascii=False, indent=2)

print(f"Saved NER answers to {ner_ids_output_dir}")
