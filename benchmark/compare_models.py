import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset  # ← THÊM Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import time
import os
import matplotlib.pyplot as plt
from utils.preprocess import normalize_text
from utils.dataset import PhoBERTIntentDataset

#  CẤU HÌNH 
DATA_PATH = "data/intents.csv"
PHOBERT_MODEL_PATH = "models/phobert_intent.pth"
BILSTM_MODEL_PATH = "models/bilstm_intent.pth"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 64

print(f"Sử dụng thiết bị: {DEVICE}")

#  LOAD DATA 
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].apply(normalize_text)

label_encoder = joblib.load(LABEL_ENCODER_PATH)
df["label"] = label_encoder.transform(df["intent"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
test_dataset = PhoBERTIntentDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

results = {}

#  ĐÁNH GIÁ PHOBERT 
print("\n=== ĐÁNH GIÁ PHOBERT ===")
phobert_model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base", num_labels=len(label_encoder.classes_)
)
phobert_model.load_state_dict(torch.load(PHOBERT_MODEL_PATH, map_location=DEVICE))
phobert_model.to(DEVICE)
phobert_model.eval()

phobert_preds = []
phobert_times = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        start = time.time()
        outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.time()
        phobert_times.append(end - start)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        phobert_preds.extend(preds)

phobert_acc = accuracy_score(y_test, phobert_preds)
phobert_report = classification_report(y_test, phobert_preds, target_names=label_encoder.classes_, output_dict=True)
phobert_infer = (sum(phobert_times) / len(phobert_times)) * 1000 / BATCH_SIZE
phobert_size = os.path.getsize(PHOBERT_MODEL_PATH) / (1024 * 1024)

results["PhoBERT"] = {
    "accuracy": round(phobert_acc, 4),
    "precision_macro": round(phobert_report['macro avg']['precision'], 4),
    "recall_macro": round(phobert_report['macro avg']['recall'], 4),
    "f1_macro": round(phobert_report['macro avg']['f1-score'], 4),
    "f1_weighted": round(phobert_report['weighted avg']['f1-score'], 4),
    "inference_time_ms": round(phobert_infer, 2),
    "model_size_mb": round(phobert_size, 2)
}

# ĐÁNH GIÁ BILSTM
print("\n=== ĐÁNH GIÁ BILSTM ===")
if not os.path.exists(BILSTM_MODEL_PATH):
    print("Không tìm thấy model BiLSTM → bỏ qua")
else:
    try:
        from training.train_bilstm import BiLSTMModel, Vocab

        # Recreate vocab đúng như khi train
        vocab = Vocab(df["text"].tolist())
        print(f"Vocab size khi load BiLSTM: {vocab.vocab_size}")

        bilstm_model = BiLSTMModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            num_classes=len(label_encoder.classes_)
        )
        bilstm_model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location=DEVICE))
        bilstm_model.to(DEVICE)
        bilstm_model.eval()

        # Dataset riêng cho BiLSTM 
        class BiLSTMIntentDataset(Dataset):
            def __init__(self, texts, labels, vocab, max_len=32):
                self.texts = texts
                self.labels = labels
                self.vocab = vocab
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoded = self.vocab.encode(self.texts[idx])
                if len(encoded) > self.max_len:
                    encoded = encoded[:self.max_len]
                else:
                    encoded += [0] * (self.max_len - len(encoded))
                return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

        bilstm_test_dataset = BiLSTMIntentDataset(X_test.tolist(), y_test.tolist(), vocab)
        bilstm_test_loader = DataLoader(bilstm_test_dataset, batch_size=BATCH_SIZE)

        bilstm_preds = []
        bilstm_times = []

        with torch.no_grad():
            for texts, labels in bilstm_test_loader:
                texts = texts.to(DEVICE)

                start = time.time()
                outputs = bilstm_model(texts)
                end = time.time()
                bilstm_times.append(end - start)

                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                bilstm_preds.extend(preds)

        bilstm_acc = accuracy_score(y_test, bilstm_preds)
        bilstm_report = classification_report(y_test, bilstm_preds, target_names=label_encoder.classes_, output_dict=True)
        bilstm_infer = (sum(bilstm_times) / len(bilstm_times)) * 1000 / BATCH_SIZE
        bilstm_size = os.path.getsize(BILSTM_MODEL_PATH) / (1024 * 1024)

        results["BiLSTM"] = {
            "accuracy": round(bilstm_acc, 4),
            "precision_macro": round(bilstm_report['macro avg']['precision'], 4),
            "recall_macro": round(bilstm_report['macro avg']['recall'], 4),
            "f1_macro": round(bilstm_report['macro avg']['f1-score'], 4),
            "f1_weighted": round(bilstm_report['weighted avg']['f1-score'], 4),
            "inference_time_ms": round(bilstm_infer, 2),
            "model_size_mb": round(bilstm_size, 2)
        }

        print(f"Accuracy: {bilstm_acc:.4f}")
        print(f"F1-macro: {bilstm_report['macro avg']['f1-score']:.4f}")
        print(f"Inference time: {bilstm_infer:.2f} ms/câu")
        print(f"Model size: {bilstm_size:.2f} MB")

    except Exception as e:
        print(f"Lỗi khi đánh giá BiLSTM: {e}")

# IN BẢNG & LƯU ẢNH 
os.makedirs("benchmark", exist_ok=True)

print("\n" + "="*80)
print("BẢNG SO SÁNH HIỆU SUẤT PHOBERT VS BILSTM")
print("="*80)

data = [
    ["Accuracy", results["PhoBERT"]["accuracy"], results.get("BiLSTM", {}).get("accuracy", "N/A")],
    ["Precision (macro)", results["PhoBERT"]["precision_macro"], results.get("BiLSTM", {}).get("precision_macro", "N/A")],
    ["Recall (macro)", results["PhoBERT"]["recall_macro"], results.get("BiLSTM", {}).get("recall_macro", "N/A")],
    ["F1-score (macro)", results["PhoBERT"]["f1_macro"], results.get("BiLSTM", {}).get("f1_macro", "N/A")],
    ["F1-score (weighted)", results["PhoBERT"]["f1_weighted"], results.get("BiLSTM", {}).get("f1_weighted", "N/A")],
    ["Inference time (ms/câu)", results["PhoBERT"]["inference_time_ms"], results.get("BiLSTM", {}).get("inference_time_ms", "N/A")],
    ["Model size (MB)", results["PhoBERT"]["model_size_mb"], results.get("BiLSTM", {}).get("model_size_mb", "N/A")],
]

for row in data:
    print(f"{row[0]:<30} {row[1]:<20} {row[2]}")

# Lưu thành ảnh
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data, colLabels=["Chỉ số", "PhoBERT", "BiLSTM"], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)

plt.savefig("benchmark/comparison_table.png", bbox_inches='tight', dpi=300)
print("\nĐã lưu bảng so sánh thành ảnh: benchmark/comparison_table.png")
print("Chèn ảnh này vào báo cáo là siêu đẹp!")

print("\nBenchmark hoàn tất! Chúc bạn bảo vệ đồ án thành công!")