import torch
from torch.utils.data.dataloader import DataLoader
from typing import Optional

def train(model: torch.nn.Module, optimizer, loss_fn, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], epochs: int = 100, save_path: str = '../models/best_model.pt'):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_dataloader:

            optimizer.zero_grad()
            y_pred = model(X_batch.permute(0, 2, 1))
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_dataloader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.inference_mode():
                for X_val, y_val in val_dataloader:
                    y_pred_val = model(X_val.permute(0, 2, 1))
                    loss_val = loss_fn(y_pred_val, y_val)
                    epoch_val_loss += loss_val.item()
    
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

        torch.save(save_path)

        # Print progress
        if (epoch + 1) % 10 == 0:
            # print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f if val_dataloader else '-'}")

    print("Training complete.")