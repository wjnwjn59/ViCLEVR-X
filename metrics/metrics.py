import torch


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy.item()
