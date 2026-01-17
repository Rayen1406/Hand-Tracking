import cv2
import mediapipe as mp
import math
import time
import numpy as np
import os
import glob

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# --- Gallery Loading ---
GALLERY_PATH = "gallery"
loaded_images = [] 
image_files = sorted(glob.glob(os.path.join(GALLERY_PATH, "*")))

for f in image_files:
    img = cv2.imread(f)
    if img is not None:
        loaded_images.append(img)
        
current_img_idx = 0

# --- Gesture Helpers ---
def calc_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_hand_state(landmarks):
    """
    Returns 'PINCHED' or 'OPEN' based on finger dispersion.
    """
    tips_ids = [4, 8, 12, 16, 20]
    tips = [landmarks.landmark[i] for i in tips_ids]
    
    cx = sum([p.x for p in tips]) / 5
    cy = sum([p.y for p in tips]) / 5
    centroid = type('obj', (object,), {'x': cx, 'y': cy})
    
    max_d = 0
    for p in tips:
        d = calc_dist(p, centroid)
        if d > max_d: max_d = d
        
    if max_d < 0.05: return 'PINCHED' # Tight pinch
    if max_d > 0.15: return 'OPEN'    # Fingers spread
    return 'NEUTRAL'

def is_two_fingers_up(landmarks):
    wrist = landmarks.landmark[0]
    def is_extended(tip_id, pip_id):
        # Tip further from wrist than PIP
        return calc_dist(landmarks.landmark[tip_id], wrist) > calc_dist(landmarks.landmark[pip_id], wrist) + 0.02
        
    index_up = is_extended(8, 6)
    middle_up = is_extended(12, 10)
    ring_up = is_extended(16, 14)
    pinky_up = is_extended(20, 18)
    
    # Strict 2 fingers: Index & Middle UP, Ring & Pinky DOWN
    return index_up and middle_up and (not ring_up) and (not pinky_up)

# --- State ---
menu_open = False
state_lock_time = 0

# Swipe State Machine
# States: 'IDLE', 'TRACKING', 'COOLDOWN'
swipe_state = 'IDLE'
swipe_start_x = 0
swipe_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Interaction Logic ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 1. Menu Toggle (Pinch = Open, Unpinch/Open = Close)
        if time.time() > state_lock_time:
            hand_state = get_hand_state(hand_landmarks)
            
            # To ensure we don't accidentally close while swiping, check swipe state?
            # Or just rely on "Two fingers up" != "Open Hand". 
            # Open hand requires ALL fingers spread. Two fingers requires Ring/Pinky down.
            # So interactions should be distinct.
            
            if hand_state == 'PINCHED' and not menu_open:
                menu_open = True
                state_lock_time = time.time() + 1.0 # 1s cooldown
            elif hand_state == 'OPEN' and menu_open:
                 # Only quit if not in the middle of a valid swipe tracking
                 if swipe_state == 'IDLE':
                    menu_open = False
                    state_lock_time = time.time() + 1.0

        # 2. Swipe Navigation (Robust State Machine)
        if menu_open and is_two_fingers_up(hand_landmarks):
            # Track Center X
            ix = hand_landmarks.landmark[8].x * w
            mx = hand_landmarks.landmark[12].x * w
            curr_x = (ix + mx) / 2
            
            if swipe_state == 'IDLE':
                swipe_state = 'TRACKING'
                swipe_start_x = curr_x
                swipe_start_time = time.time()
                
            elif swipe_state == 'TRACKING':
                dx = curr_x - swipe_start_x
                dt = time.time() - swipe_start_time
                
                # Check Distance Threshold (Commitment)
                if abs(dx) > 40:  # Lowered threshold for higher sensitivity
                    # Check Speed (Intent)
                    if dt < 0.4: # Increased time window to allow slightly slower swipes
                        if dx > 0: # Right (L->R) -> Next
                             if current_img_idx < len(loaded_images) - 1:
                                 current_img_idx += 1
                                 cv2.circle(frame, (w-50, h//2), 30, (0, 200, 0), -1) # Highlight Right
                        else: # Left (R->L) -> Prev
                             if current_img_idx > 0:
                                 current_img_idx -= 1
                                 cv2.circle(frame, (50, h//2), 30, (0, 200, 0), -1) # Highlight Left
                        
                        swipe_state = 'COOLDOWN' # Triggered success
                    else:
                        # Too slow (Drifting) -> Invalid
                        swipe_state = 'COOLDOWN' 
            
            # If COOLDOWN, we wait here doing nothing until fingers dropped
                        
        else:
            # Fingers dropped or changed pose -> Reset
            swipe_state = 'IDLE'

    # --- Rendering ---
    if menu_open:
        # Overlay
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        if loaded_images:
            img = loaded_images[current_img_idx]
            ih, iw, _ = img.shape
            
            # Target Size
            target_w = 500
            scale = target_w / iw
            nh, nw = int(ih * scale), int(iw * scale)
            
            # Clamp
            if nh > h - 100:
                scale = (h - 100) / ih
                nh, nw = int(ih * scale), int(iw * scale)
                
            img_resized = cv2.resize(img, (nw, nh))
            dy = (h - nh) // 2
            dx = (w - nw) // 2
            
            frame[dy:dy+nh, dx:dx+nw] = img_resized
            
            # Hints (Arrows)
            l_center_y = h // 2
            
            # Left Arrow (Prev) - Show only if we have prev images
            if current_img_idx > 0:
                cv2.arrowedLine(frame, (100, l_center_y), (40, l_center_y), (255, 255, 255), 5, tipLength=0.5)
            
            # Right Arrow (Next) - Show only if we have next images
            if current_img_idx < len(loaded_images) - 1:
                cv2.arrowedLine(frame, (w-100, l_center_y), (w-40, l_center_y), (255, 255, 255), 5, tipLength=0.5)
            
            # Counter
            cv2.putText(frame, f"{current_img_idx+1}/{len(loaded_images)}", 
                        (w//2 - 40, dy + nh + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Text Hint
            cv2.putText(frame, "Open hand to close", (w//2 - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
             cv2.putText(frame, "No Images", (w//2 - 80, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
    else:
        # Hint text when closed
        cv2.putText(frame, "Pinch 5 fingers to open menu", (20, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
    # Debug State
    # cv2.putText(frame, f"State: {swipe_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Hand Tracking Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()