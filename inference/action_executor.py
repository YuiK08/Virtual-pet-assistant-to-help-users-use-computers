import os
import subprocess
import keyboard

# Import hàm chung từ pet_window 
from inference.pet_window import set_pet_state

# MỞ / ĐÓNG TRÌNH DUYỆT & YOUTUBE 
def open_browser():
    set_pet_state("push")
    subprocess.Popen(["start", "chrome"], shell=True)
    print(" Đang mở Google Chrome...")

def close_browser():
    set_pet_state("push")
    os.system("taskkill /f /im chrome.exe >nul 2>&1")
    print(" Đã đóng tất cả cửa sổ Chrome!")

def open_youtube():
    set_pet_state("push")
    subprocess.Popen(['start', 'chrome', 'https://www.youtube.com'], shell=True)
    print(" Đang mở YouTube...")

# ÂM LƯỢNG
def volume_up():
    set_pet_state("push")
    keyboard.press_and_release('volume up')
    print(" Âm lượng đã tăng!")

def volume_down():
    set_pet_state("push")
    keyboard.press_and_release('volume down')
    print(" Âm lượng đã giảm!")

#  PHẢN HỒI THÚ CƯNG 
def greet_pet():
    set_pet_state("happy")
    print(" Woof woof! Chào chủ nhân yêu quý ơi~ ")

def praise_pet():
    set_pet_state("happy")
    print(" Hihi cảm ơn chủ nhân! Thú cưng vui lắm luôn~ ")

def pet_sleep():
    set_pet_state("sleep")
    print(" Thú cưng đi ngủ đây... Zzz... Ngủ ngon nha chủ nhân ")

def pet_minimize():
    set_pet_state("sleep")  
    print(" Thú cưng thu nhỏ và chui xuống góc chờ lệnh...")

# THỰC THI 
def execute_action(intent: str):
    actions = {
        "open_browser": open_browser,
        "close_browser": close_browser,
        "open_youtube": open_youtube,
        "volume_up": volume_up,
        "volume_down": volume_down,
        "greet_pet": greet_pet,
        "praise_pet": praise_pet,
        "pet_sleep": pet_sleep,
        "pet_minimize": pet_minimize
    }
    
    func = actions.get(intent)
    if func:
        func()
    else:
        print(f" Chưa biết làm gì với lệnh '{intent}'. Chủ nhân dạy thêm đi nha!")