import cv2
import numpy as np
import pyautogui
import mss
import time
from tkinter import messagebox, Tk
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

class AspectRatioWidget(QMainWindow):
    def __init__(self, width, height, aspect_ratio):
        super().__init__()
        self.aspect_ratio = aspect_ratio
        self.setWindowTitle('Screen Capture')
        
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        
        self.resize(width, height)
        self.setMinimumSize(width//2, height//2)
        
    def resizeEvent(self, event):
        new_size = event.size()
        desired_height = int(new_size.width() / self.aspect_ratio)
        
        if desired_height > new_size.height():
            desired_width = int(new_size.height() * self.aspect_ratio)
            self.resize(desired_width, new_size.height())
        else:
            self.resize(new_size.width(), desired_height)
        
        super().resizeEvent(event)

root = Tk()
root.withdraw()
use_gpu = messagebox.askyesno("GPU Acceleration", "Do you want to use GPU acceleration?")
root.destroy()

if use_gpu:
    cv2.ocl.setUseOpenCL(True)
    print("GPU acceleration enabled")
else:
    cv2.ocl.setUseOpenCL(False)
    print("GPU acceleration disabled")

screen_width, screen_height = pyautogui.size()
scale_factor = 2
aspect_ratio = screen_width / screen_height
window_width = screen_width // scale_factor
window_height = int(window_width / aspect_ratio)

print("============================")
print(f"Monitor Width: {screen_width}")
print(f"Monitor Height: {screen_height}")
print(f"App Windows Width: {window_width}")
print(f"App Windows Height: {window_height}")
print("============================")

sct = mss.mss()
monitor = sct.monitors[1]

bounding_box = {
    "top": monitor["top"],
    "left": monitor["left"],
    "width": monitor["width"],
    "height": monitor["height"]
}

original_cursor_img = cv2.imread('cursor.png', cv2.IMREAD_UNCHANGED)
cursor_height = 16
cursor_aspect_ratio = original_cursor_img.shape[1] / original_cursor_img.shape[0]
cursor_width = int(cursor_height * cursor_aspect_ratio)
cursor_img = cv2.resize(original_cursor_img, (cursor_width, cursor_height))

if cursor_img.shape[2] == 4:
    cursor_alpha = cursor_img[:, :, 3] / 255.0
    cursor_rgb = cursor_img[:, :, :3]

fps = 0
fps_time = time.time()
frame_count = 0

def main():
    global frame_count, fps, fps_time
    
    app = QApplication([])
    window = AspectRatioWidget(window_width, window_height, aspect_ratio)
    window.show()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 255, 0)
    text_position = (10, 20)

    def update_frame():
        global frame_count, fps, fps_time
        
        screenshot = np.array(sct.grab(bounding_box))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        current_width = window.width()
        current_height = window.height()
        
        frame = cv2.resize(frame, (current_width, current_height), 
                         interpolation=cv2.INTER_NEAREST)
        
        current_mouse_x, current_mouse_y = pyautogui.position()
        scaled_mouse_x = int(current_mouse_x / scale_factor)
        scaled_mouse_y = int(current_mouse_y / scale_factor)
        
        if cursor_img.shape[2] == 4:
            if 0 <= scaled_mouse_y < current_height - cursor_height and \
               0 <= scaled_mouse_x < current_width - cursor_width:
                roi = frame[scaled_mouse_y:scaled_mouse_y+cursor_height, 
                          scaled_mouse_x:scaled_mouse_x+cursor_width]
                if roi.shape[:2] == cursor_rgb.shape[:2]:
                    for c in range(3):
                        roi[:, :, c] = roi[:, :, c] * (1 - cursor_alpha) + \
                                     cursor_rgb[:, :, c] * cursor_alpha
        
        current_time = time.time()
        if current_time - fps_time >= 1:
            fps = frame_count / (current_time - fps_time)
            fps_time = current_time
            frame_count = 0
        
        cv2.putText(frame, f'FPS: {int(fps)}', text_position, 
                    font, font_scale, font_color, font_thickness)
        
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        window.label.setPixmap(QPixmap.fromImage(q_img))
        
        frame_count += 1

    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(16)  
    
    app.exec_()

if __name__ == '__main__':
    main()