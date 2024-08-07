import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class VQADataset(Dataset):
    """
    A custom Dataset class for Visual Question Answering (VQA) tasks.

    This class processes and prepares data for VQA models, including images, questions, answers, and explanations.

    Args:
        data (dict): A dictionary containing the VQA data.
        image_dir (str): Path to the directory containing images.
        transform (callable, optional): A function/transform to apply to the images.
        max_question_length (int, optional): Maximum length of questions. Defaults to 20.
        max_explanation_length (int, optional): Maximum length of explanations. Defaults to 50.
        max_vocab_size (int, optional): Maximum size of the vocabulary. Defaults to 10000.

    Attributes:
        data (dict): The VQA data.
        image_dir (str): Path to the image directory.
        transform (callable): Image transformation function.
        question_ids (list): List of question IDs.
        max_question_length (int): Maximum question length.
        max_explanation_length (int): Maximum explanation length.
        max_vocab_size (int): Maximum vocabulary size.
        word2idx (dict): Mapping of words to indices.
        idx2word (dict): Mapping of indices to words.
        answer2idx (dict): Mapping of answers to indices.
        idx2answer (dict): Mapping of indices to answers.
    """
    def __init__(self, data, image_dir, transform=None, max_question_length=20, max_explanation_length=50, max_vocab_size=10000):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.question_ids = list(self.data.keys())
        self.max_question_length = max_question_length
        self.max_explanation_length = max_explanation_length
        self.max_vocab_size = max_vocab_size
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.answer2idx = {}
        self.idx2answer = {}
        
        self.build_vocab()
        self.build_answer_vocab()
        
    def build_vocab(self):
        """
        Builds the vocabulary from questions and explanations in the dataset.
        """
        word_freq = Counter()
        for item in self.data.values():
            word_freq.update(word_tokenize(item['question'].lower()))
            word_freq.update(word_tokenize(item.get('explanation', [''])[0].lower()))
        
        for word, _ in word_freq.most_common(self.max_vocab_size - 4):  # -4 for <PAD>, <UNK>, <START>, <END>
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_answer_vocab(self):
        """
        Builds the answer vocabulary from the dataset.
        """
        answer_freq = Counter()
        for item in self.data.values():
            answers = [ans['answer'].lower() for ans in item['answers']]
            answer_freq.update(answers)
        
        for answer, _ in answer_freq.most_common():
            idx = len(self.answer2idx)
            self.answer2idx[answer] = idx
            self.idx2answer[idx] = answer
    
    def tokenize(self, text):
        """
        Tokenizes and converts text to token indices.

        Args:
            text (str): Input text to tokenize.

        Returns:
            list: List of token indices.
        """
        tokens = word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def pad_sequence(self, sequence, max_length):
        """
        Pads or truncates a sequence to a fixed length.

        Args:
            sequence (list): Input sequence.
            max_length (int): Desired length of the sequence.

        Returns:
            list: Padded or truncated sequence.
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))
    
    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.question_ids)
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the processed image, question, answer, explanation, and question ID.
        """
        question_id = self.question_ids[idx]
        item = self.data[question_id]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Process question
        question = self.tokenize(item['question'])
        question = self.pad_sequence(question, self.max_question_length)
        question = torch.LongTensor(question)
        
        # Process answers
        answers = [ans['answer'].lower() for ans in item['answers']]
        answer_count = Counter(answers)
        most_common_answer = answer_count.most_common(1)[0][0]
        answer = self.answer2idx.get(most_common_answer, 0)
        
        # Process explanation
        explanation = item.get('explanation', [''])[0]
        explanation = self.tokenize(explanation)
        explanation = [self.word2idx['<START>']] + explanation + [self.word2idx['<END>']]
        explanation = self.pad_sequence(explanation, self.max_explanation_length)
        explanation = torch.LongTensor(explanation)
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'explanation': explanation,
            'question_id': question_id
        }

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def process_data(train_path, val_path, test_path, train_image_dir, val_image_dir, test_image_dir, batch_size=512, num_workers=4):
    """
    Processes VQA data and creates DataLoader objects for training, validation, and testing.

    Args:
        train_path (str): Path to the training data JSON file.
        val_path (str): Path to the validation data JSON file.
        test_path (str): Path to the test data JSON file.
        train_image_dir (str): Path to the directory containing training images.
        val_image_dir (str): Path to the directory containing validation images.
        test_image_dir (str): Path to the directory containing test images.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 512.
        num_workers (int, optional): Number of worker processes for DataLoaders. Defaults to 4.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - word2idx (dict): Mapping of words to indices.
            - answer2idx (dict): Mapping of answers to indices.
            - idx2word (dict): Mapping of indices to words.
            - idx2answer (dict): Mapping of indices to answers.
    """
    # Load datasets
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    
    # Create custom dataset classes
    train_dataset = VQADataset(train_data, train_image_dir)
    val_dataset = VQADataset(val_data, val_image_dir)
    test_dataset = VQADataset(test_data, test_image_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.word2idx, train_dataset.answer2idx, train_dataset.idx2word, train_dataset.idx2answer

if __name__ == "__main__":
    # Example usage
    # File paths
    train_path = "/home/VLAI/datasets/VQA-X/vqaX_train.json"
    val_path = "/home/VLAI/datasets/VQA-X/vqaX_val.json"
    test_path = "/home/VLAI/datasets/VQA-X/vqaX_test.json"
    train_image_dir = '/home/VLAI/datasets/COCO_Images/train2014'
    val_image_dir = '/home/VLAI/datasets/COCO_Images/val2014'
    test_image_dir = '/home/VLAI/datasets/COCO_Images/val2014'

    # Process data and create data loaders
    train_loader, val_loader, test_loader, word2idx, answer2idx, idx2word, idx2answer = process_data(
        train_path, val_path, test_path, train_image_dir, val_image_dir, test_image_dir)

    # iterating through the data loader
    for batch in train_loader:
        print("Batch of images:", batch['image'].shape)
        print("Batch of questions:", batch['question'].shape)
        print("Batch of answers:", batch['answer'].shape)
        print("Batch of explanations:", batch['explanation'].shape)
        print("Batch of question IDs:", batch['question_id'])
        break