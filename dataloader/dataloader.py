import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
import os
from PIL import Image
from dotenv import load_dotenv
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

load_dotenv()

train_path = os.environ.get("train_path")
val_path = os.environ.get("val_path")
image_folder = os.environ.get("train_images")


class VQADataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        # Load dataset from CSV
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        # Initialize the tokenizer
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.data['question']), specials=["<unk>"])
        
        # Preprocess the dataset
        self.preprocess_dataset()

    def preprocess_dataset(self):
        dataset = load_dataset(
            "csv",
            data_files={
                "train": train_path,
                "test": val_path
            }
        )

        with open("/home/VLAI/minhth/ViCLEVR-X/datasets/dataset/answer_space.txt") as f:
            answer_space = f.read().splitlines()

        self.data = dataset.map(
            lambda examples: {
                'label': [
                    answer_space.index(ans.replace(" ", "").split(",")[0])  
                    for ans in examples['answer']
                ]
            },
            batched=True
        )

        # Convert dataset to DataFrame for easy indexing
        self.data = pd.DataFrame(self.data['train'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        question = row['question']
        label = row['label']

        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Tokenize the question
        tokenized_question = self.tokenizer(question)
        # Ensure the tensors are in the correct format for the DataLoader
        input_ids = torch.tensor(self.vocab(tokenized_question), dtype=torch.long)  # Remove batch dimension
        attention_mask = torch.ones_like(input_ids)  # Remove batch dimension

        return image, input_ids, attention_mask, label