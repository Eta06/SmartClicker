import time
import cv2
import numpy as np
import pyautogui
from mss import mss
import threading

# Load and resize the cursor image
original_cursor_img = cv2.imread('cursor.png', cv2.IMREAD_UNCHANGED)
cursor_height = 24
aspect_ratio = original_cursor_img.shape[1] / original_cursor_img.shape[0]
cursor_width = int(cursor_height * aspect_ratio)
cursor_img = cv2.resize(original_cursor_img, (cursor_width, cursor_height), interpolation=cv2.INTER_AREA)

# Pre-compute alpha blending for cursor
cursor_alpha = cursor_img[:, :, 3] / 255.0
cursor_alpha = cursor_alpha[:, :, np.newaxis]
cursor_rgb = cursor_img[:, :, :3]

# Get screen resolution
screen_width, screen_height = pyautogui.size()
scale_factor = 2
window_width = screen_width // scale_factor
window_height = screen_height // scale_factor

print("============================")
print(f"Monitor Width: {screen_width}")
print(f"Monitor Height: {screen_height}")
print(f"App Windows Width: {window_width}")
print(f"App Windows Height: {window_height}")
print("============================")

# Global variables for threading
frame = None
mouse_x, mouse_y = 0, 0
running = True

# Create a lock for thread-safe operations
lock = threading.Lock()


def capture_screen():
    global frame, mouse_x, mouse_y, running

    # Create a new mss instance for this thread
    with mss() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
        fps_start_time = time.time()
        fps_counter = 0
        fps_display = "FPS: 0"
        fps_update_interval = 0.5  # Update FPS every 0.5 seconds

        while running:
            # Capture the screen
            img = np.array(sct.grab(monitor))

            # Resize the frame
            resized_frame = cv2.resize(img, (window_width, window_height))

            # Get mouse cursor position
            current_mouse_x, current_mouse_y = pyautogui.position()

            # Update global variables in a thread-safe manner
            with lock:
                frame = resized_frame
                mouse_x, mouse_y = current_mouse_x, current_mouse_y

            # FPS Calculation
            fps_counter += 1
            if (time.time() - fps_start_time) > fps_update_interval:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_display = f"FPS: {fps:.2f}"
                fps_counter = 0
                fps_start_time = time.time()

            # Display FPS on the frame
            cv2.putText(resized_frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def display_screen():
    global frame, mouse_x, mouse_y, running
    while running:
        with lock:
            if frame is not None:
                display_frame = frame.copy()
                current_mouse_x, current_mouse_y = mouse_x, mouse_y

        if frame is not None:
            # Scale cursor position
            cursor_x = int(current_mouse_x * window_width / screen_width) - cursor_width // 2
            cursor_y = int(current_mouse_y * window_height / screen_height) - cursor_height // 2

            # Ensure cursor is within frame bounds
            if 0 <= cursor_x < window_width - cursor_width and 0 <= cursor_y < window_height - cursor_height:
                # Apply pre-computed alpha blending
                roi = display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3]
                blended = (1 - cursor_alpha) * roi + cursor_alpha * cursor_rgb
                display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3] = blended.astype(
                    np.uint8)

            # Display the frame
            cv2.imshow("Screen", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break


# Start the capture and display threads
capture_thread = threading.Thread(target=capture_screen)
display_thread = threading.Thread(target=display_screen)

capture_thread.start()
display_thread.start()

# Wait for threads to finish
capture_thread.join()
display_thread.join()

cv2.destroyAllWindows()