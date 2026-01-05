import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from utils.dataset import PhoBERTIntentDataset
from utils.preprocess import normalize_text

# Cấu hình
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Load data
df = pd.read_csv("data/intents.csv")
df["text"] = df["text"].apply(normalize_text)

# Encode label
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])

# Lưu label encoder
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Split train/val
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset & DataLoader
train_dataset = PhoBERTIntentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
val_dataset = PhoBERTIntentDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_encoder.classes_)
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/phobert_intent.pth")
        print("Đã lưu model tốt nhất!")

print(f"\nHuấn luyện hoàn tất! Accuracy tốt nhất: {best_acc:.4f}")