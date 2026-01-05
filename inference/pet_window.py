import pygame
import threading
import time
import os

# CẤU HÌNH 
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "assets")

SPRITESHEETS = {
    "idle": "idle.png",
    "push": "push.png",
    "happy": "happy.png",
    "sleep": "sleep.png"
}

NUM_FRAMES = {
    "idle": 12,
    "push": 12,
    "happy": 12,
    "sleep": 12
}

FRAME_WIDTH = 192
FRAME_HEIGHT = 192
DISPLAY_SCALE = 2

# Biến toàn cục
frames = {}
current_state = "idle"
current_frame_idx = 0
state_timer = time.time()
animation_speed = 8

# Hiệu ứng minimize
is_minimizing = False
minimize_start_time = 0
minimize_duration = 1.5  # giây

# LOAD 
def load_frames():
    global frames
    for state, filename in SPRITESHEETS.items():
        path = os.path.join(ASSETS_PATH, filename)
        if not os.path.exists(path):
            print(f"Không tìm thấy {filename}")
            continue

        sheet = pygame.image.load(path).convert_alpha()
        print(f"Load {state}: {sheet.get_size()}")

        frame_list = []
        for i in range(NUM_FRAMES[state]):
            x = i * FRAME_WIDTH
            rect = pygame.Rect(x, 0, FRAME_WIDTH, FRAME_HEIGHT)
            try:
                frame = sheet.subsurface(rect)
                scaled = pygame.transform.scale(frame, (FRAME_WIDTH * DISPLAY_SCALE, FRAME_HEIGHT * DISPLAY_SCALE))
                frame_list.append(scaled)
            except Exception as e:
                print(f"Lỗi cắt frame {i} của {state}: {e}")
                break
        frames[state] = frame_list
        print(f"{state}: {len(frame_list)} frame")

# CỬA SỔ 
def run_pet_window():
    global current_state, current_frame_idx, state_timer, is_minimizing, minimize_start_time

    pygame.init()
    original_size = (FRAME_WIDTH * DISPLAY_SCALE, FRAME_HEIGHT * DISPLAY_SCALE)
    screen = pygame.display.set_mode(original_size, pygame.NOFRAME)
    pygame.display.set_caption("Thú cưng ảo")

    load_frames()

    # Trong suốt + luôn trên cùng
    try:
        hwnd = pygame.display.get_wm_info()["window"]
        import ctypes
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x80000)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 200, 200, 0, 0, 0x0001)
    except:
        pass

    clock = pygame.time.Clock()
    running = True
    frame_timer = 0

    # Vị trí ban đầu
    current_pos = [200, 200]
    current_size = original_size

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_timer += 1
        if frame_timer >= animation_speed:
            frame_timer = 0
            if current_state in frames and frames[current_state]:
                current_frame_idx = (current_frame_idx + 1) % len(frames[current_state])

        screen.fill((0, 0, 0, 0))

        current_frames = frames.get(current_state) or frames.get("idle", [])
        if current_frames:
            frame = current_frames[current_frame_idx]

            # Hiệu ứng thu nhỏ xuống góc phải khi minimize
            if is_minimizing:
                elapsed = time.time() - minimize_start_time
                progress = min(elapsed / minimize_duration, 1.0)

                # Ease out
                t = 1 - (1 - progress) ** 3

                # Vị trí: từ hiện tại đến góc phải dưới
                start_x, start_y = current_pos
                target_x = pygame.display.get_desktop_sizes()[0][0] - current_size[0] - 50
                target_y = pygame.display.get_desktop_sizes()[0][1] - current_size[1] - 50
                current_pos[0] = start_x + (target_x - start_x) * t
                current_pos[1] = start_y + (target_y - start_y) * t

                # Scale từ 1 xuống 0.3
                scale = 1 - 0.7 * t
                new_width = int(original_size[0] * scale)
                new_height = int(original_size[1] * scale)
                scaled_frame = pygame.transform.scale(frame, (new_width, new_height))

                screen.blit(scaled_frame, current_pos)

                if progress >= 1.0:
                    is_minimizing = False
            else:
                screen.blit(frame, (0, 0))

        pygame.display.flip()
        clock.tick(60)

        # Reset về idle sau 4 giây (trừ sleep)
        if current_state not in ["sleep"] and time.time() - state_timer > 4:
            current_state = "idle"
            current_frame_idx = 0  # Reset frame để idle animation chạy lại từ đầu

    pygame.quit()

#  ĐIỀU KHIỂN 
def set_pet_state(state):
    global current_state, state_timer, current_frame_idx
    if state in frames:
        current_state = state
        current_frame_idx = 0  # Reset frame để animation chạy lại từ đầu
        state_timer = time.time()

def pet_push():     set_pet_state("push")
def pet_happy():    set_pet_state("happy")
def pet_sleep():    set_pet_state("sleep")

def pet_minimize():
    global is_minimizing, minimize_start_time
    set_pet_state("sleep")
    is_minimizing = True
    minimize_start_time = time.time()

# Khởi động
threading.Thread(target=run_pet_window, daemon=True).start()
time.sleep(1.5)