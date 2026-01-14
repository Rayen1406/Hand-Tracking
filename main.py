import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Read one frame to get dimensions
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
    max_crop_size = min(h, w)
else:
    max_crop_size = 400 # Fallback

prev_crop_size = max_crop_size
prev_cx, prev_cy = 0, 0 # Initialize with 0 or center. Center is better safely set in loop if w/h known, but 0 is safe start.
# Better to init to center if we can, but we might not have w,h yet reliably.
# Let's check if we have w,h from line 17. 
if ret:
    prev_cx, prev_cy = w // 2, h // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    # Update dimensions in case they change/init fail
    h, w, _ = frame.shape
    max_crop_size = min(h, w)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    zoomed_frame = frame.copy()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Thumb tip & index tip
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]

        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)

        # Midpoint of pinch
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Distance between fingers
        distance = math.hypot(x2 - x1, y2 - y1)

        if distance > 30:
            # Zoom in as distance increases? Code logic was:
            # 500 - distance*2. If distance = 30, val=440. If dist=200, val=100.
            # So scaling from max_crop_size down to 100.
            
            # Map distance [30, 200] to crop size [max_crop_size, 100]
            zoom_factor = 2.5
            target_size = max_crop_size - (distance - 30) * zoom_factor
            new_crop_size = int(max(100, min(max_crop_size, target_size)))
        else:
            new_crop_size = max_crop_size  # Full view

        # EXPERIMENTAL SMOOTHING
        alpha = 0.1 # Lower = smoother but more lag (0.1 is very smooth)
        
        # Smooth Zoom
        crop_size = int(prev_crop_size * (1 - alpha) + new_crop_size * alpha)
        prev_crop_size = crop_size
        
        # Smooth Position (Panning)
        # Apply smoothing to cx, cy
        smoothed_cx = int(prev_cx * (1 - alpha) + cx * alpha)
        smoothed_cy = int(prev_cy * (1 - alpha) + cy * alpha)
        prev_cx, prev_cy = smoothed_cx, smoothed_cy
        
        # Use smoothed center for crop bounds
        center_x, center_y = smoothed_cx, smoothed_cy

        # Crop area around smoothed midpoint
        x_start = max(0, center_x - crop_size//2)
        y_start = max(0, center_y - crop_size//2)
        x_end = min(w, center_x + crop_size//2)
        y_end = min(h, center_y + crop_size//2)

        crop = frame[y_start:y_end, x_start:x_end]

        # Resize crop to full window
        zoomed_frame = cv2.resize(crop, (w, h))

        # Actual crop dimensions (might be smaller due to clamping)
        actual_crop_h, actual_crop_w, _ = crop.shape

        # Convert hand coordinates to mapped zoomed coordinates
        # Use actual_crop_w/h to be precise
        scale_x = w / actual_crop_w if actual_crop_w > 0 else 1
        scale_y = h / actual_crop_h if actual_crop_h > 0 else 1
        
        rel_x1 = int((x1 - x_start) * scale_x)
        rel_y1 = int((y1 - y_start) * scale_y)
        rel_x2 = int((x2 - x_start) * scale_x)
        rel_y2 = int((y2 - y_start) * scale_y)


        # Draw the line and circles using relative coordinates
        cv2.circle(zoomed_frame, (rel_x1, rel_y1), 10, (255, 0, 0), -1)
        cv2.circle(zoomed_frame, (rel_x2, rel_y2), 10, (255, 0, 0), -1)
        cv2.line(zoomed_frame, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 255, 0), 3)
        cv2.putText(zoomed_frame, f"Zoom Distance: {distance:.0f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Zoom Camera", zoomed_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()