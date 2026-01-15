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
loaded_images = [] # Tuples: (original, thumbnail)
image_files = sorted(glob.glob(os.path.join(GALLERY_PATH, "*")))

for f in image_files:
    img = cv2.imread(f)
    if img is not None:
        # Resize for thumbnail (fixed width 120, maintain aspect ratio)
        h, w, _ = img.shape
        thumb_w = 120
        thumb_h = int(h * (thumb_w / w))
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        loaded_images.append((img, thumb))

# --- UI Helpers ---
def draw_rounded_rect(img, rect, color, radius=15, alpha=1.0, filled=True, progress_fill=0.0):
    """
    Draws a rounded rectangle with alpha blending and optional progress fill.
    progress_fill: 0.0 to 1.0 (fills background from left to right)
    """
    x, y, w, h = rect
    
    # Create overlay
    overlay = img.copy()
    
    # 1. Background Fill
    if filled:
        # Check boundary to avoid drawing errors
        if radius * 2 > w or radius * 2 > h: radius = min(w, h) // 2

        # Draw Base Rounded Rect
        cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color, -1)
        cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color, -1)
        cv2.circle(overlay, (x+radius, y+radius), radius, color, -1)
        cv2.circle(overlay, (x+w-radius, y+radius), radius, color, -1)
        cv2.circle(overlay, (x+w-radius, y+h-radius), radius, color, -1)
        cv2.circle(overlay, (x+radius, y+h-radius), radius, color, -1)
        
        # Progress Fill (Green overlay on top of button color)
        if progress_fill > 0:
            fill_w = int(w * progress_fill)
            if fill_w > 0:
                # To account for rounded corners properly is hard, simplified: 
                # Fill from left. 
                cv2.rectangle(overlay, (x, y), (x + fill_w, y + h), (0, 255, 0), -1)

    # Blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_centered_text(img, text, rect, font_scale=0.8, color=(255, 255, 255), thickness=2):
    x, y, w, h = rect
    (fw, fh), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = x + (w - fw) // 2
    text_y = y + (h + fh) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# --- State ---
hover_start_time = None
thumb_hover_start = None
thumb_hover_idx = -1

is_expanded = False
scroll_offset = 0
preview_idx = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- UI Layout ---
    btn_w, btn_h = 200, 80
    btn_x = w - btn_w - 40
    btn_y = 50
    btn_rect = (btn_x, btn_y, btn_w, btn_h)
    
    panel_h = 400
    panel_y = btn_y + btn_h + 10
    panel_rect = (btn_x, panel_y, btn_w, panel_h)
    
    # --- Interaction Logic ---
    cursor_pos = None
    is_hovering_btn = False
    btn_progress = 0
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        index_tip = hand_landmarks.landmark[8]
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        cursor_pos = (ix, iy)
        
        # 1. Preview Mode Interaction (Close)
        if preview_idx is not None:
            close_btn_rect = (w//2 - 100, h - 100, 200, 60)
            if (close_btn_rect[0] <= ix <= close_btn_rect[0]+200) and (close_btn_rect[1] <= iy <= close_btn_rect[1]+60):
                if hover_start_time is None: hover_start_time = time.time()
                elapsed = time.time() - hover_start_time
                if elapsed > 1.5:
                     preview_idx = None
                     hover_start_time = None
            else:
                 # Only reset if we are NOT on button. But inside preview mode loop below, 
                 # we don't have else logic easily.
                 # Simplified: 
                 pass 
        
        # 2. Main Menu Interaction (Only if not in preview)
        elif (btn_rect[0] <= ix <= btn_rect[0]+btn_w) and (btn_rect[1] <= iy <= btn_rect[1]+btn_h):
            is_hovering_btn = True
            if hover_start_time is None: hover_start_time = time.time()
            elapsed = time.time() - hover_start_time
            btn_progress = min(elapsed / 2.0, 1.0)
            if elapsed > 2.0:
                is_expanded = not is_expanded
                hover_start_time = None
        
        # 3. Panel Interaction (Scroll & Thumbs)
        elif is_expanded and (panel_rect[0] <= ix <= panel_rect[0]+btn_w) and (panel_rect[1] <= iy <= panel_rect[1]+panel_h):
            # Scroll Zones
            rel_y = iy - panel_y
            if rel_y < 60: # Top 60px -> Scroll Up
                scroll_offset = max(0, scroll_offset - 10)
            elif rel_y > panel_h - 60: # Bottom 60px -> Scroll Down
                # Allow infinite scroll down for simplicity or check len(loaded_images)
                scroll_offset += 10
            
            # Thumbnail Hover
            content_y = panel_y + 40 - scroll_offset
            hovered_idx = -1
            
            for i, (orig, thumb) in enumerate(loaded_images):
                th, tw, _ = thumb.shape
                roi_x = btn_x + (btn_w - tw)//2
                
                if (roi_x <= ix <= roi_x+tw) and (content_y <= iy <= content_y+th):
                    if (content_y >= panel_y) and (content_y + th <= panel_y + panel_h):
                        hovered_idx = i
                
                content_y += th + 20
            
            if hovered_idx != -1:
                if thumb_hover_idx != hovered_idx:
                    thumb_hover_idx = hovered_idx
                    thumb_hover_start = time.time()
                elif time.time() - thumb_hover_start > 1.0: # 1s to open preview
                    preview_idx = hovered_idx
                    thumb_hover_idx = -1
                    thumb_hover_start = None
            else:
                thumb_hover_idx = -1
                thumb_hover_start = None

        else:
            hover_start_time = None
            thumb_hover_idx = -1

    # --- Rendering ---
    
    # 1. Preview Mode Overlay
    if preview_idx is not None:
        # Dark Overlay
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Large Image
        img_full = loaded_images[preview_idx][0]
        ih, iw, _ = img_full.shape
        scale = min((w - 100)/iw, (h - 200)/ih) # Fit to screen with padding
        nh, nw = int(ih*scale), int(iw*scale)
        if nw > 0 and nh > 0:
            img_resized = cv2.resize(img_full, (nw, nh))
            dx = (w - nw)//2
            dy = (h - nh)//2
            frame[dy:dy+nh, dx:dx+nw] = img_resized
        
        # Close Button
        close_x = w//2 - 100
        close_y = h - 100
        close_rect = (close_x, close_y, 200, 60)
        
        # Check hover for progress drawing
        close_prog = 0
        if cursor_pos: 
            cx, cy = cursor_pos
            if (close_x <= cx <= close_x+200) and (close_y <= cy <= close_y+60):
                if hover_start_time:
                    close_prog = min((time.time() - hover_start_time)/1.5, 1.0)
        
        draw_rounded_rect(frame, close_rect, (0, 0, 255), 30, 0.8, filled=True, progress_fill=close_prog)
        draw_centered_text(frame, "CLOSE", close_rect)
        
        # Draw Cursor
        if cursor_pos: cv2.circle(frame, cursor_pos, 8, (255, 0, 255), -1)

    else:
        # 2. Collapsing Panel
        if is_expanded:
            draw_rounded_rect(frame, panel_rect, (255, 255, 255), 15, 0.2)
            
            # Content
            current_y = panel_y + 40 - scroll_offset
            
            for i, (orig, thumb) in enumerate(loaded_images):
                th, tw, _ = thumb.shape
                roi_x = btn_x + (btn_w - tw)//2
                roi_y = int(current_y)
                
                # Manual Clipping
                if roi_y >= panel_y and roi_y + th <= panel_y + panel_h:
                     frame[roi_y:roi_y+th, roi_x:roi_x+tw] = thumb
                     
                     # Highlight if hovering
                     if i == thumb_hover_idx:
                         cv2.rectangle(frame, (roi_x, roi_y), (roi_x+tw, roi_y+th), (0, 255, 255), 2)
                         # Progress bar on top of thumb?
                         if thumb_hover_start:
                             prog = (time.time() - thumb_hover_start) / 1.0
                             cv2.rectangle(frame, (roi_x, roi_y+th-5), (roi_x+int(tw*prog), roi_y+th), (0, 255, 0), -1)

                current_y += th + 20

        # 3. Main Menu Button
        btn_alpha = 0.4 if is_hovering_btn else 0.2
        draw_rounded_rect(frame, btn_rect, (255, 255, 255), 20, btn_alpha, filled=True, progress_fill=btn_progress)
        text = "MENU" if not is_expanded else "CLOSE"
        draw_centered_text(frame, text, btn_rect, 1.0)
        
        # Cursor
        if cursor_pos:
             cv2.circle(frame, cursor_pos, 8, (255, 0, 255), -1)

    cv2.imshow("Hand Tracking Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()