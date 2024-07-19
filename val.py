import torch.nn as nn
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.dataloader import VQADataset
from transformers import BertModel
from metrics.metrics import calculate_accuracy
from tqdm import tqdm


class VQAModel_trained(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel_trained, self).__init__()
        # Image feature extractor
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        # Question feature extractor
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion and final classification
        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = self.cnn(images)
        
        # Extract question features
        outputs = self.bert(input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate image and question features
        combined = torch.cat((image_features, question_features), dim=1)
        
        # classify
        x = self.fc1(combined)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
        for images, input_ids, attention_mask, labels in progress_bar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += accuracy

    loss = running_loss / len(data_loader)
    accuracy = running_accuracy / len(data_loader)

    print(f"Loss: {loss}, Accuracy: {accuracy}")

    return loss, accuracy

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQAModel_trained(num_answers=582).to(device)
    
    # Load the best model checkpoint 
    checkpoint_path = "./models/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = VQADataset("/home/minhth/VQA-baseline/ViCLEVR-X/datasets/dataset/data_eval.csv", "/home/minhth/VQA-baseline/ViCLEVR-X/datasets/dataset/images", transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Test data loader prepared successfully.")
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
            