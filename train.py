import torch
from torch.optim import AdamW, lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from model import model
from dataset import train_loader, val_loader
from model import FrequencyAwareLoss

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# unfreeze 'head' and 'layers.3'
for name, param in model.named_parameters():
    if name.startswith("head"):
        param.requires_grad = True
    elif name.startswith("layers.3"):
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Fine-tune 1 stage (layers.3 + head) | Trainable params: {trainable}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 4
learning_rate = 1e-4

model = model.to(device)
# criterion = nn.CrossEntropyLoss()
criterion = FrequencyAwareLoss(ce_weight=1.0, ft_weight=0.5).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
scaler = GradScaler()

best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    pred_correct = 0
    total_samples = 0

    for batch_imgs, batch_labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
    ):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

        optimizer.zero_grad()  # clear gradients from previous step
        with autocast(device.type):
            logits = model(batch_imgs)  # shape: [batch_size, num_classes]
            loss = criterion(logits, batch_labels, batch_imgs)  # Compute loss

        scaler.scale(loss).backward()  # Backward pass: compute gradients

        scaler.unscale_(optimizer)
        # Gradient clipping prevents exploding gradients (common in transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update model weights

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * batch_imgs.size(0)
        predicted = torch.argmax(logits, dim=1)
        pred_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    scheduler.step()

    train_loss /= total_samples
    train_acc = pred_correct / total_samples

    model.eval()
    val_loss = 0.0
    pred_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(val_loader, desc="  Validation:"):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

            with autocast(device.type):
                logits = model(batch_imgs)
                loss_val = criterion(logits, batch_labels, batch_imgs)
                probs = torch.softmax(logits, dim=1)

            val_loss += loss_val.item() * batch_imgs.size(0)
            predicted = torch.argmax(logits, dim=1)
            pred_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    val_loss /= total_samples
    val_acc = pred_correct / total_samples

    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("----Model saved----")
