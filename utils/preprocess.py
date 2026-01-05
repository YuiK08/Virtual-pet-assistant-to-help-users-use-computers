import re
import unicodedata

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Chuẩn hóa unicode (rất quan trọng cho tiếng Việt)
    text = unicodedata.normalize("NFC", text)
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, giữ chữ cái, số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text