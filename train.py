# train.py (revisi: shape-safe untuk ResNet-18)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import DenseNet121Binary
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train():
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)

    model = DenseNet121Binary(in_channels=in_channels, pretrained=True).to(DEVICE)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    train_losses_history, val_losses_history = [], []
    train_accs_history, val_accs_history = [], []

    print("\n--- Memulai Training DenseNet-121 ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            # Pastikan ke device dan tipe float
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()  # shape: (batch_size, 1)

            outputs = model(images)              # shape: (batch_size, 1)
            outputs = outputs.squeeze(1)         # shape -> (batch_size,)
            labels_flat = labels.squeeze(1)      # shape -> (batch_size,)

            # Debug shapes (uncomment jika mau cek)
            # print("DEBUG train shapes:", outputs.shape, labels_flat.shape)

            loss = criterion(outputs, labels_flat)  # both shape (batch,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()  # shape (batch,)
            train_total += labels_flat.size(0)
            train_correct += (preds == labels_flat).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # VALIDATION
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float()  # (batch,1)

                outputs = model(images)             # (batch,1)
                outputs = outputs.squeeze(1)        # (batch,)
                labels_flat = labels.squeeze(1)     # (batch,)

                # Debug shapes (uncomment if needed)
                # print("DEBUG val shapes:", outputs.shape, labels_flat.shape)

                val_loss = criterion(outputs, labels_flat)
                val_running_loss += val_loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels_flat.size(0)
                val_correct += (preds == labels_flat).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        scheduler.step()

        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("\n--- Training Selesai ---")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/densenet-121.pth")
    print("Saved model to saved_models/densenet-121_final.pth")

    # Plot and visualize
    plot_training_history(train_losses_history, val_losses_history,
                         train_accs_history, val_accs_history)
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()