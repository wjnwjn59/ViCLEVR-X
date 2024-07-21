import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torch.utils.checkpoint import checkpoint


class VQAModel(nn.Module):
    def __init__(self, num_answers, vocab_size, embed_dim=256):
        super(VQAModel, self).__init__()
        # Image feature extractor
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer

        # Question feature extractor
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fusion and final classification
        self.fc1 = nn.Linear(2048 + 512, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = checkpoint(self.cnn, images)  # Use checkpointing for ResNet

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
        return x
