import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# Define paths
dataset_dir = "../../../datasets/VQA-X"
sampling_dir = os.path.join(dataset_dir, "final_data/sampling")
output_dir = os.path.join(dataset_dir, "final_data/post_processing")
os.makedirs(output_dir, exist_ok=True)
ner_ids_output_dir = os.path.join(output_dir, "ner_answers.json")
item_per_batch = 5

# Set up CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

# Load pre-trained BERT-based NER model and tokenizer
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize NER pipeline with GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

ner_answers = []

# Keywords for special translation cases
special_translation_keywords = {
    "American": "Mỹ",
    "polar": "Bắc Cực",
    "polar bear": "gấu Bắc Cực",
    "microwave": "lò vi sóng",
    "dalmatian": "chó đốm",
    "leopard": "báo",
    "fox": "cáo",
    "crosstown": "đi ngang qua",
    "desert": "sa mạc",
    "Oxford Circus": "Oxford Circus",
    "Addington Village": "Làng Addington",
}


def get_most_common_answer(answers):
    """Returns the most common answer from the list of answers."""
    return pd.Series([answer["answer"] for answer in answers]).mode()[0]


def contains_named_entity(text):
    """Check if the text contains a named entity using the NER pipeline."""
    ner_results = ner_pipeline(text)
    return any(ent["entity"].startswith(("B-", "I-")) for ent in ner_results)


def translate_special_cases(answer):
    """Translate special cases like 'America' to 'Mỹ' based on context."""
    for keyword, translation in special_translation_keywords.items():
        if keyword.lower() in answer.lower():
            return translation
    return answer


# Main loop to process datasets
for dataset in ["train", "val", "test"]:
    # Load original and sampling data
    with open(os.path.join(dataset_dir, f"vqaX_{dataset}.json"), "r") as f:
        original_data = json.load(f)
    sampling_data = pd.read_csv(
        os.path.join(sampling_dir, f"{dataset}_sampling_final.csv")
    )

    for start_index in tqdm(
        range(0, len(sampling_data), item_per_batch), desc=f"Processing {dataset}"
    ):
        samples_batch = []

        # Collect batch of samples
        for index in range(
            start_index, min(start_index + item_per_batch, len(sampling_data))
        ):
            sample = sampling_data.iloc[index]
            question_id = str(sample["question_id"])
            original_question = original_data[question_id]["question"]
            original_answer = get_most_common_answer(
                original_data[question_id]["answers"]
            )
            img_id = original_data[question_id]["image_id"]

            samples_batch.append(
                {
                    "question": original_question,
                    "answer": original_answer,
                    "index": index,  # Store index for later use in sampling_data
                    "question_id": question_id,  # Store question ID for NER tracking
                    "img_id": img_id,  # Store image ID
                }
            )

        if samples_batch:
            for sample in samples_batch:
                question = sample["question"]
                answer = sample["answer"]

                # Check if there is a named entity in the answer
                if contains_named_entity(answer):
                    # Handle special cases like 'America' -> 'Mỹ'
                    translated_answer = translate_special_cases(answer)
                    if translated_answer != answer:
                        sampling_data.at[sample["index"], "answer"] = translated_answer
                    else:
                        # Keep the original answer if it's a named entity and not a special case
                        sampling_data.at[sample["index"], "answer"] = answer
                        ner_answers.append(answer)

                # Add image ID to the sampling data
                sampling_data.at[sample["index"], "img_id"] = sample["img_id"]

    # Save the processed dataset
    sampling_data.to_csv(
        os.path.join(output_dir, f"vqaX_{dataset}_translated.csv"), index=False
    )

# Save the list of NER question IDs to a JSON file for future reference
with open(ner_ids_output_dir, "w") as ner_file:
    json.dump(ner_answers, ner_file, ensure_ascii=False, indent=2)

print(f"Saved NER question IDs to {ner_ids_output_dir}")
