class Config:
    num_epochs = 100
    lr = 0.0005
    weight_decay = 1e-4
    best_loss = float('inf')
    best_model_state = None
    patience = 10  # Number of epochs to wait for improvement before stopping
    early_stopping_counter = 0
