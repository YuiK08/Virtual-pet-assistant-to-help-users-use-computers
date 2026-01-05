# training/train_bilstm.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time
from utils.preprocess import normalize_text

# ==================== CẤU HÌNH ====================
DATA_PATH = "data/intents.csv"
MODEL_SAVE_PATH = "models/bilstm_intent.pth"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
MAX_LEN = 32  # Câu ngắn, giảm padding thừa
DROPOUT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")

# ==================== XÂY DỰNG VOCAB ====================
class Vocab:
    def __init__(self, texts):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        freq = {}
        for text in texts:
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
        
        for word in sorted(freq, key=freq.get, reverse=True):
            if len(self.word2idx) >= 10000:  # Giới hạn vocab
                break
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.word2idx) - 1] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text):
        return [self.word2idx.get(word, 1) for word in text.split()]

# ==================== DATASET ====================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.vocab.encode(text)
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded += [0] * (self.max_len - len(encoded))

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ==================== MODEL BILSTM ====================
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.bilstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Kết hợp 2 chiều
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# ==================== LOAD DATA ====================
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].apply(normalize_text)

# Label encoder (dùng chung với PhoBERT để so sánh công bằng)
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    df["label"] = label_encoder.transform(df["intent"])
else:
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Xây dựng vocab
vocab = Vocab(df["text"].tolist())

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

train_dataset = IntentDataset(X_train.tolist(), y_train.tolist(), vocab)
val_dataset = IntentDataset(X_val.tolist(), y_val.tolist(), vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==================== HUẤN LUYỆN ====================
model = BiLSTMModel(
    vocab_size=vocab.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=len(label_encoder.classes_)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0
print("\nBắt đầu huấn luyện BiLSTM...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            preds.extend(torch.argmax(outputs, dim=-1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  → Đã lưu model tốt nhất (Acc: {acc:.4f})")

print(f"\nHuấn luyện hoàn tất! Best Val Accuracy: {best_acc:.4f}")
print(f"Model lưu tại: {MODEL_SAVE_PATH}")
print("Giờ bạn có thể chạy benchmark/compare_models.py để so sánh với PhoBERT!")