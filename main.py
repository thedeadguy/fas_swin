import cv2 as cv
import numpy as np
import timm
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"
OPTIMAL_THRESHOLD = 0.9617

IMAGENET_MEAN = [0.485, 0.467, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
NUM_CLASSES = 2

print(f"Loading model on {DEVICE}....")

model_name = "swin_tiny_patch4_window7_224"
model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()
print("Model Architecture: Swin-Tiny Transformer")

img_tranform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

mtcnn = MTCNN(keep_all=True, device=DEVICE)

print("Starting webcam.... Press 'q' to quit")
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("http://10.197.16.240:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    boxes, pobs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            h, w = frame.shape[:2]

            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)

            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_tensor = img_tranform(face_crop).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = torch.softmax(logits, dim=1)
                    # Spoof (index 1)
                    spoof_prob = probs[0][1].item()

                print(f"Raw spoof_prob: {spoof_prob}")
                if spoof_prob >= OPTIMAL_THRESHOLD:
                    label = "SPOOF"
                    color = (0, 0, 255)
                else:
                    label = "REAL"
                    color = (0, 255, 0)

                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display_text = f"{label} ({spoof_prob * 100:.1f}%)"
                cv.putText(
                    frame,
                    display_text,
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

    cv.imshow("FAS SWIN", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
