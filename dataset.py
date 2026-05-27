from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


BASE_DIR = "/content/LCC_FASD"
TRAIN_DIR = os.path.join(BASE_DIR, "LCC_FASD_training")
VAL_DIR = os.path.join(BASE_DIR, "LCC_FASD_development")
TEST_DIR = os.path.join(BASE_DIR, "LCC_FASD_evaluation")

IMAGENET_MEAN = [0.485, 0.467, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
test_dataset = ImageFolder(root=TEST_DIR, transform=val_transform)
class_names = train_dataset.classes

print(f"Class names: {class_names}")
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")
print(f"Total test samples: {len(test_dataset)}")

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# import torch
# import matplotlib.pyplot as plt

# # verify image after normalization inversion
# def denormalize(img_tensor):
#   mean = torch.tensor(IMAGENET_MEAN).view(3, 1,1)
#   std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
#   return torch.clamp(img_tensor * std + mean, 0, 1)

# sample_img, sample_labels = next(iter(train_loader))

# fig, axes = plt.subplots(2,4, figsize=(14,7))
# fig.suptitle("Sample from Traning Batch (LCC_FASD)",fontsize=14,fontweight='bold')
# for i, ax in enumerate(axes.flat):
#   if i >= 8:
#     break

#   img = denormalize(sample_img[i]).permute(1,2,0).numpy()
#   label_name = class_names[sample_labels[i].item()]
#   color = 'green' if label_name == 'real' else 'red'
#   ax.imshow(img)
#   ax.set_title(label_name, color=color, fontweight='bold')
#   ax.axis('off')

# plt.tight_layout()
# plt.show()
