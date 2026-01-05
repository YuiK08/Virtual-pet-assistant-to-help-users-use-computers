# inference/rule_engine.py

def rule_based_predict(text: str):
    """
    Rule-based intent detection với ưu tiên cao để xử lý các lệnh phổ biến và dễ nhầm.
    Ưu tiên rule này hơn PhoBERT để đảm bảo độ chính xác gần 100% cho các lệnh cơ bản.
    """
    if not text:
        return None
    
    # Chuẩn hóa để rule dễ bắt hơn (giống preprocess nhưng giữ nguyên từ khóa chính)
    t = text.lower().strip()
    # 1. MỞ TRÌNH DUYỆT / YOUTUBE (ưu tiên cao nhất vì hay bị nhầm)
    open_verbs = ["mở", "mo", "bật", "bat", "vào", "vao", "khởi động", "khoi dong", "cho xem", "xem"]
    if any(verb in t for verb in open_verbs):
        if any(obj in t for obj in ["chrome", "trình duyệt", "trinh duyet", "browser", "web", "cốc cốc", "coc coc", "google"]):
            return "open_browser"
        if any(obj in t for obj in ["youtube", "yt", "you tube"]):
            return "open_youtube"
    # 2. ĐÓNG TRÌNH DUYỆT
    close_verbs = ["đóng", "dong", "tắt", "tat", "thoát", "thoat", "close", "đóng lại", "dong lai"]
    if any(verb in t for verb in close_verbs):
        if any(obj in t for obj in ["chrome", "trình duyệt", "trinh duyet", "browser", "tab", "cửa sổ", "cua so"]):
            return "close_browser"
    # 3. ÂM LƯỢNG
    if any(word in t for word in ["tăng", "tang", "to", "mở to", "mo to", "vặn to", "van to", "lớn tiếng", "lon tieng"]):
        if any(word in t for word in ["âm lượng", "am luong", "volume", "tiếng", "tieng", "loa"]):
            return "volume_up"

    if any(word in t for word in ["giảm", "giam", "nhỏ", "nho", "hạ", "ha", "vặn nhỏ", "van nho", "im"]):
        if any(word in t for word in ["âm lượng", "am luong", "volume", "tiếng", "tieng", "loa"]):
            return "volume_down"
    # 4. CHÀO HỎI / KHEN THÚ CƯNG
    greet_words = ["chào", "chao", "hi", "hello", "xin chào", "xin chao", "ê", "e", "alo", "hey"]
    if any(word in t for word in greet_words):
        return "greet_pet"

    praise_words = ["giỏi", "gioi", "ngoan", "dễ thương", "de thuong", "thông minh", "thong minh", "xuất sắc", "xuat sac", "siêu", "sieus", "tốt lắm", "tot lam"]
    if any(word in t for word in praise_words):
        return "praise_pet"

    # 5. THÚ CƯNG NGỦ / THU NHỎ
    sleep_words = ["ngủ", "ngu", "nghỉ", "nghi", "dừng", "dung", "tạm dừng", "tam dung", "đi ngủ", "di ngu", "ngủ đi", "ngu di"]
    if any(word in t for word in sleep_words):
        return "pet_sleep"

    minimize_words = ["thu nhỏ", "thu nho", "ẩn", "an", "chui xuống", "chui xuong", "trốn", "tron", "ẩn đi", "an di", "xuống góc", "xuong goc"]
    if any(word in t for word in minimize_words):
        return "pet_minimize"

    # 6. TẠM BIỆT 
    goodbye_words = ["tạm biệt", "tam biet", "bye", "hẹn gặp lại", "hen gap lai", "ngủ ngon", "ngu ngon"]
    if any(word in t for word in goodbye_words):
        return "greet_pet"  
    return None