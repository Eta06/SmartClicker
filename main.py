import time
import cv2
import numpy as np
import pyautogui
from mss import mss
import threading

# Ask the user if they want to use OpenCL
use_gpu = input("Use GPU acceleration with OpenCL (y/n)? ").lower() == 'y'

if use_gpu:
    try:
        import pyopencl as cl
        import pyopencl.array

        platforms = cl.get_platforms()
        if len(platforms) == 0:
            raise cl.RuntimeError("No OpenCL platforms found")

        platform = platforms[0]  # Choose the first platform
        devices = platform.get_devices(device_type=cl.device_type.ALL)
        if len(devices) == 0:
            raise cl.RuntimeError("No OpenCL devices found")

        device = devices[0]  # Choose the first device
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        print(f"Using OpenCL device: {device.name}")
    except Exception as e:
        print(f"OpenCL initialization failed: {str(e)}")
        print("Falling back to CPU")
        use_gpu = False

# Load the cursor image
original_cursor_img = cv2.imread('cursor.png', cv2.IMREAD_UNCHANGED)
cursor_height = 16  # Smaller cursor size
aspect_ratio = original_cursor_img.shape[1] / original_cursor_img.shape[0]
cursor_width = int(cursor_height * aspect_ratio)
cursor_img = cv2.resize(original_cursor_img, (cursor_width, cursor_height), interpolation=cv2.INTER_AREA)

# Pre-compute alpha blending for cursor
cursor_alpha = cursor_img[:, :, 3] / 255.0
cursor_alpha = cursor_alpha[:, :, np.newaxis]
cursor_rgb = cursor_img[:, :, :3]

if use_gpu:
    cursor_alpha_gpu = cl.array.to_device(queue, cursor_alpha.astype(np.float32))
    cursor_rgb_gpu = cl.array.to_device(queue, cursor_rgb.astype(np.float32))

    # OpenCL kernel for alpha blending
    blend_kernel = cl.Program(context, """
    __kernel void alpha_blend(__global const float4 *bg, __global const float *alpha, 
                                    __global const float4 *fg, __global float4 *out) {
        int gid = get_global_id(0);
        float a = alpha[gid];
        out[gid] = (1-a)*bg[gid] + a*fg[gid];
    }
    """).build()

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
fps_display = "FPS: 0"  # Initialize FPS display

# Create a lock for thread-safe operations
lock = threading.Lock()

def capture_screen():
    global frame, mouse_x, mouse_y, running, fps_display

    with mss() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
        fps_start_time = time.time()
        fps_counter = 0
        fps_update_interval = 0.5  # Update FPS every 0.5 seconds

        while running:
            img = np.array(sct.grab(monitor))
            resized_frame = cv2.resize(img, (window_width, window_height))
            current_mouse_x, current_mouse_y = pyautogui.position()

            with lock:
                frame = resized_frame
                mouse_x, mouse_y = current_mouse_x, current_mouse_y

            # FPS Calculation
            fps_counter += 1
            if (time.time() - fps_start_time) > fps_update_interval:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_display = f"FPS: {fps:.2f}"  # Update the global FPS display
                fps_counter = 0
                fps_start_time = time.time()


def display_screen():
    global frame, mouse_x, mouse_y, running, fps_display
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
                if use_gpu:
                    roi = display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3]
                    roi_gpu = cl.array.to_device(queue, roi.astype(np.float32))
                    out_gpu = cl.array.empty_like(roi_gpu)

                    blend_kernel.alpha_blend(queue, roi.shape[:2], None, roi_gpu.data,
                                            cursor_alpha_gpu.data, cursor_rgb_gpu.data, out_gpu.data)

                    blended = out_gpu.get()
                    display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3] = blended.astype(
                        np.uint8)
                else:
                    # Apply pre-computed alpha blending on CPU
                    roi = display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3]
                    blended = (1 - cursor_alpha) * roi + cursor_alpha * cursor_rgb
                    display_frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, :3] = blended.astype(
                        np.uint8)

            # Make the FPS counter smaller and less intrusive
            cv2.putText(display_frame, fps_display, (10, 15),  # Positioned at the top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Smaller font size and thickness

            cv2.imshow("Screen", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# Start the capture and display threads
capture_thread = threading.Thread(target=capture_screen)
display_thread = threading.Thread(target=display_screen)

capture_thread.start()
display_thread.start()

capture_thread.join()
display_thread.join()

cv2.destroyAllWindows()