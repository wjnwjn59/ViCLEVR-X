import torch
import wandb
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from dotenv import load_dotenv
from dataloader.dataloader import VQADataset
from models.baseline import VQAModel
from config.model_cfg import Config
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from metrics.metrics import calculate_accuracy
import json

load_dotenv()

wandb_key = os.environ.get("WANDB")
train_path = os.environ.get("train_path")
train_path_image = os.environ.get("train_image")
val_path = os.environ.get("val_path")
val_path_image = os.environ.get("val_image")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def createDataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    train_dataset = VQADataset(train_path, train_path_image, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = VQADataset(val_path, val_path_image, transform)
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scaler, device, num_epochs, patience, save_path, project_name):
    # Initialize wandb
    wandb.init(project=project_name)
    wandb.watch(model, log="all")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, input_ids, attention_mask, labels in progress_bar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += accuracy

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        wandb.log({"train_loss": epoch_loss,
                  "train_accuracy": epoch_accuracy, "epoch": epoch+1})

        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0

        with torch.no_grad():
            val_progress_bar = tqdm(
                val_loader, desc="Validating", unit="batch")
            for images, input_ids, attention_mask, labels in val_progress_bar:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                val_running_accuracy += accuracy

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_running_accuracy / len(val_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        wandb.log(
            {"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch+1})

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}")
        print(
            f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        # Check if the validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            early_stopping_counter = 0  # Reset counter if we get a new best loss
            print(f"Saving model with lowest validation loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, save_path)
        else:
            early_stopping_counter += 1
            print(
                f"No improvement in validation loss for {early_stopping_counter} epochs.")

        # Check for early stopping
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save the final metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }

    wandb.finish()

    return metrics


if __name__ == "__main__":
    set_seed(42)
    wandb.login(key=wandb_key, relogin=True)

    # Initialize the model
    model = VQAModel(num_answers=582)
    model.to(device)
    # Training parameters
    num_epochs = Config.num_epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    best_loss = Config.best_loss
    best_model_state = Config.best_model_state
    patience = Config.patience  # Number of epochs to wait for improvement before stopping
    early_stopping_counter = Config.early_stopping_counter

    # Initialize the optimizer and GradScaler for mixed precision training
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    scaler = GradScaler()

    # Define the loss function
    criterion = CrossEntropyLoss()

    # Define scheduler
    scheduler_step_size = int(num_epochs * 0.25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_step_size)

    # Dataset
    train_loader, val_loader = createDataset()

    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scaler, device,
                          num_epochs, patience, "/home/minhth/VQA-baseline/ViCLEVR-X/models/best_model.pth", "VQA_BASELINE(ResNet50-Bert)")
    with open("/home/minhth/VQA-baseline/ViCLEVR-X/metrics/metrics.json", "w") as f:
        json.dump(metrics, f)
    pass
