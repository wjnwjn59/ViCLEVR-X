import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceGeneration
from torchvision import models
from torch.utils.checkpoint import checkpoint
from timm import create_model

class VQAModel(nn.Module):
    def __init__(self, num_answers, vocab_size, embed_dim=256):
        super(VQAModel, self).__init__()
        # Image feature extractor
        self.vit = create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the final classification layer

        # Question feature extractor
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion and final classification
        self.fc1 = nn.Linear(768 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.3)

        # Initialize tokenizer and model for generating explanations
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.explanation_generator = BertForSequenceGeneration.from_pretrained('bert-base-uncased')  # Placeholder, should be fine-tuned or replaced with a better model

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = checkpoint(self.vit, images)  # Use checkpointing for ResNet

        # Extract question features
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        question_features = lstm_output[:, -1, :]
        
        # Concatenate features
        combined_features = torch.cat((image_features, question_features), dim=1)

        # Classification
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)

        # Generate explanations
        generated_explanations = []
        for i in range(images.size(0)):
            image_chunk = images[i].unsqueeze(0)  # Assuming batch size is 1 for simplicity
            question_chunk = input_ids[i].unsqueeze(0)
            explanation = self.generate_explanation(image_chunk, question_chunk)
            generated_explanations.append(explanation)

        return x, generated_explanations

    def generate_explanation(self, image, question):
        inputs = self.tokenizer(question, image, return_tensors='pt', padding=True, truncation=True)
        outputs = self.explanation_generator(**inputs)
        generated_tokens = torch.argmax(outputs.logits, dim=-1)
        explanation = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return explanation
