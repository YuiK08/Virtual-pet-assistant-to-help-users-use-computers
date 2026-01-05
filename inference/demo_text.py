import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from utils.preprocess import normalize_text
from inference.rule_engine import rule_based_predict
from inference.action_executor import execute_action  

# LOAD MODEL & TOKENIZER 
print("Đang tải mô hình PhoBERT và label encoder...")

# Load label encoder để biết số lớp
label_encoder = joblib.load("models/label_encoder.pkl")
num_labels = len(label_encoder.classes_)

# Load tokenizer và model với đúng số lớp
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=num_labels
)

# Load trọng số đã train
model.load_state_dict(torch.load("models/phobert_intent.pth", map_location=torch.device('cpu')))
model.eval()

print("Mô hình đã sẵn sàng!\n")

#  CHÀO MỪNG & VÒNG LẶP DEMO 
print(" TRỢ LÝ THÚ CƯNG ẢO ĐÃ HOẠT ĐỘNG THẬT SỰ TRÊN MÁY TÍNH! ")
print("Bạn có thể ra lệnh bằng tiếng Việt (có dấu hoặc không dấu đều được)")
print("Gõ 'bye', 'thoát', 'tạm biệt' hoặc 'ngủ đi' để kết thúc\n")

while True:
    try:
        text = input("Bạn: ").strip()
    except KeyboardInterrupt:
        print("\n Thú cưng: Bye bye chủ nhân!")
        break

    # Kết thúc chương trình
    if text.lower() in ["bye", "thoát", "tạm biệt", "ngủ đi", "sleep", "tam biet"]:
        print(" Thú cưng: Bye bye chủ nhân yêu quý! Đi ngủ đây~  Zzz...")
        break

    if not text:
        print(" Hãy nói gì đó với thú cưng đi nào~")
        continue

    # Tiền xử lý văn bản
    cleaned = normalize_text(text)

    #  ƯU TIÊN RULE-BASED 
    rule_intent = rule_based_predict(cleaned)
    if rule_intent:
        print(f" Rule-based phát hiện: {rule_intent}")
        execute_action(rule_intent)
        print()  # Dòng trống cho đẹp
        continue

    #  DÙNG PHOBERT NẾU RULE KHÔNG BẮT ĐƯỢC
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()

    intent = label_encoder.inverse_transform([pred_idx])[0]

    print(f" PhoBERT dự đoán: {intent} (độ tin cậy: {confidence:.2f})")
    execute_action(intent)
    print()  # Dòng trống cho đẹp
