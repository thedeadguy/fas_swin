import torch
import numpy as np
from sklearn.metrics import roc_curve
from torch.nn import CrossEntropyLoss
from torch.amp import autocast
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from model import model
from dataset import val_loader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
criterion = nn.CrossEntropyLoss()

model.eval()
pred_correct = 0
total_samples = 0
all_predictions = []
all_true_labels = []
all_probabilities = []
with torch.no_grad():
    for batch_imgs, batch_labels in tqdm(val_loader, desc="  Validation:"):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        with autocast(device.type):
            logits = model(batch_imgs)
            loss_val = criterion(logits, batch_labels)
            probs = torch.softmax(logits, dim=1)

        predicted = torch.argmax(logits, dim=1)
        pred_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())

val_acc = pred_correct / total_samples

f1 = f1_score(all_true_labels, all_predictions, average="macro")
probs_positive = [p[1] for p in all_probabilities]  # index 1 = 'Spoof'
auc = roc_auc_score(all_true_labels, probs_positive)

print(f"\nAccuracy: {val_acc * 100:>9.1f}")
print(f"F1 Score: {f1 * 100:>9.1f}")
print(f"AUC: {auc:>10.4f}")

# Convert to numpy arrays
y_true = np.array(all_true_labels)
y_scores = np.array(probs_positive)  # Probability of being 'Spoof'

# Generate ROC curve to sweep through all possible probability thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# In standard FAS datasets where Spoof = 1 and Real = 0:
# FPR (False Positive Rate) = Real faces misclassified as Spoof = BPCER
# FNR (False Negative Rate) = 1 - TPR = Spoof faces misclassified as Real = APCER
bpcer_array = fpr
apcer_array = 1 - tpr

# Find the EER threshold (where APCER and BPCER are as close to equal as possible)
eer_index = np.nanargmin(np.absolute((apcer_array - bpcer_array)))
eer = bpcer_array[eer_index]
optimal_threshold = thresholds[eer_index]

# Calculate metrics at this optimal threshold
best_apcer = apcer_array[eer_index]
best_bpcer = bpcer_array[eer_index]
acer = (best_apcer + best_bpcer) / 2

print("\n--- ISO/IEC 30107-3 FAS METRICS ---")
print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
print(f"EER (Equal Error Rate):        {eer * 100:.2f}%")
print(f"APCER (Attack Error Rate):     {best_apcer * 100:.2f}%")
print(f"BPCER (Bona Fide Error Rate):  {best_bpcer * 100:.2f}%")
print(f"ACER (Average Error Rate):     {acer * 100:.2f}%")
