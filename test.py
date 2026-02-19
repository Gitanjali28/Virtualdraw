import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pytesseract  # OCR

#python -m pip uninstall mediapipe -y
#python -m pip install mediapipe==0.10.9
#to install mediapipe correct version

# ---------------- TESSERACT CONFIG ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None

# Colors (more options)
colors = [
    (0,0,255),       # Red
    (0,255,0),       # Green
    (255,0,0),       # Blue
    (0,255,255),     # Yellow
    (255,0,255),     # Magenta
    (255,255,0),     # Cyan
    (255,255,255),   # White
    (0,0,0)          # Black
]
color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "White", "Black"]
selected_color = colors[0]

# Drawing state
current_stroke = []
all_strokes = []
prev_x, prev_y = None, None
is_drawing = False

# Save
save_dir = "SavedDrawings"
os.makedirs(save_dir, exist_ok=True)
save_start_time = None

# OCR
ocr_text = ""

# ---------------- FUNCTIONS ----------------
def fingers_up(hand):
    tips = [4,8,12,16,20]
    fingers = []

    # Thumb
    fingers.append(1 if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x else 0)

    for i in range(1,5):
        fingers.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)

    return fingers

# ✅ OCR FUNCTION (improved)
def get_text_from_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get drawing only
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Invert: Tesseract works best with black text on white
    thresh = cv2.bitwise_not(thresh)
    
    # OCR configuration: PSM 7 = single line, PSM 6 = block
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    return text.strip()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 🎨 Color palette
    for i, color in enumerate(colors):
        x1, y1 = 10 + i*70, 10
        x2, y2 = x1+60, 60
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, -1)
        cv2.putText(frame, color_names[i], (x1,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) if color!=(255,255,255) else (0,0,0), 1)

    # 💾 SAVE button
    cv2.rectangle(frame, (540,10),(620,60),(200,200,200),-1)
    cv2.putText(frame,"SAVE",(550,45),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(hand)
        h, w, _ = frame.shape
        index_x = int(hand.landmark[8].x * w)
        index_y = int(hand.landmark[8].y * h)

        # 🎨 Color select by hover
        for i in range(len(colors)):
            if 10+i*70 < index_x < 70+i*70 and 10 < index_y < 60:
                selected_color = colors[i]

        # 💾 SAVE with hold
        if 540 < index_x < 620 and 10 < index_y < 60:
            if save_start_time is None:
                save_start_time = time.time()
            elif time.time() - save_start_time > 0.8:
                filename = f"{save_dir}/drawing_{int(time.time())}.png"
                cv2.imwrite(filename, canvas)
                print("Saved:", filename)
                save_start_time = None
        else:
            save_start_time = None

        # 🧹 Open palm → clear
        if sum(fingers) == 5:
            canvas[:] = 0
            all_strokes.clear()
            current_stroke.clear()
            prev_x, prev_y = None, None
            ocr_text = ""  # Clear OCR text

        # ✏️ PEN DOWN (Index UP)
        if fingers[1] == 1:
            if not is_drawing:
                is_drawing = True
                current_stroke = []

            current_stroke.append((index_x, index_y))

            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y),
                         (index_x, index_y), selected_color, 8)  # Thicker for better OCR

            prev_x, prev_y = index_x, index_y

        # ✋ PEN UP (Index DOWN)
        else:
            if is_drawing:
                all_strokes.append(current_stroke.copy())
                current_stroke.clear()

                # ✅ OCR Trigger
                detected_text = get_text_from_canvas(canvas)
                if detected_text:
                    ocr_text = detected_text
                    print("Detected Text:", ocr_text)

            is_drawing = False
            prev_x, prev_y = None, None

    # Merge canvas
    frame = cv2.add(frame, canvas)
    cv2.putText(frame, f"Color: {color_names[colors.index(selected_color)]}",
                (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, selected_color, 2)

    # ✅ Display OCR text
    if ocr_text:
        cv2.putText(frame, f"OCR: {ocr_text}", (10,160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow("Air Draw - Pen Up / Pen Down + OCR", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
