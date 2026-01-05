def get_action(intent: str):
    mapping = {
        "open_browser": "Mở trình duyệt Chrome",
        "close_browser": "Đóng trình duyệt",
        "open_youtube": "Mở YouTube",
        "greet_pet": "Thú cưng vẫy đuôi chào lại",
        "praise_pet": "Thú cưng vui mừng, nhảy nhót",
        "volume_up": "Tăng âm lượng hệ thống",
        "volume_down": "Giảm âm lượng hệ thống",
        "pet_sleep": "Thú cưng đi ngủ (ẩn tạm)",
        "pet_minimize": "Thu nhỏ thú cưng xuống góc màn hình"
    }
    return mapping.get(intent, "Không hiểu lệnh")