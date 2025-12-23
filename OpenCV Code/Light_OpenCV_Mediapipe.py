import cv2
import mediapipe as mp
import math
import serial
import time

# ----------------- Setup -----------------
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Set up serial communication with Arduino
arduino = serial.Serial('/dev/cu.usbmodem11101', 9600, timeout=1)
time.sleep(2)  # allow Arduino to reset

# Flags for save mode
save_mode = False

# LED states
led_states = {
    'index': False,
    'middle': False,
    'ring': False,
    'little': False,
    'thumb': False
}

# EMA for angle smoothing
ema_alpha = 0.2
last_stable_angle = None

# Servo last angle (to keep fixed when exiting Save Mode)
servo_angle = 90  # default mid-position


# ----------------- Helper Functions -----------------
def detect_fingers(landmarks):
    """Detect which fingers are raised"""
    fingers = []
    if landmarks.landmark[8].y < landmarks.landmark[6].y:
        fingers.append("index")
    if landmarks.landmark[12].y < landmarks.landmark[10].y:
        fingers.append("middle")
    if landmarks.landmark[16].y < landmarks.landmark[14].y:
        fingers.append("ring")
    if landmarks.landmark[20].y < landmarks.landmark[18].y:
        fingers.append("little")
    if landmarks.landmark[4].x < landmarks.landmark[3].x:
        fingers.append("thumb")
    return fingers


def send_led_state():
    """Send LED states to Arduino"""
    led_string = ''.join(['1' if led_states[finger] else '0'
                          for finger in ['index', 'middle', 'ring', 'little', 'thumb']])
    arduino.write((led_string + "\n").encode())
    print(f"LED state sent: {led_string}")


def send_servo_angle(angle):
    """Send servo angle to Arduino"""
    global servo_angle
    servo_angle = int(angle)  # save last angle for Normal Mode
    msg = f"A{servo_angle}\n"
    arduino.write(msg.encode())
    print(f"Servo angle sent: {servo_angle}")


def update_leds(fingers):
    """Update LEDs in Normal Mode only"""
    global led_states, save_mode
    if save_mode:
        return  # skip LED updates

    state_changed = False
    for finger in led_states.keys():
        if (finger in fingers and not led_states[finger]) or (finger not in fingers and led_states[finger]):
            led_states[finger] = not led_states[finger]
            state_changed = True

    if state_changed:
        send_led_state()


def calculate_angle(p1, p2, p3):
    """Angle between 3 points"""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p1[0] - p3[0], p1[1] - p3[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    angle_rad = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle_rad)


def calculate_stable_angle_ema(p1, p2, p3):
    """Smooth angle with EMA"""
    global last_stable_angle
    angle = calculate_angle(p1, p2, p3)
    if last_stable_angle is None:
        last_stable_angle = angle
    else:
        last_stable_angle = ema_alpha * angle + (1 - ema_alpha) * last_stable_angle
    return last_stable_angle


# ----------------- Main Program -----------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_counter = 0
frame_skip = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = detect_fingers(landmarks)
            update_leds(fingers)

            if save_mode:
                # Get hand keypoints
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                h, w, _ = frame.shape
                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
                thumb_mcp_coords = (int(thumb_mcp.x * w), int(thumb_mcp.y * h))
                index_mcp_coords = (int(index_mcp.x * w), int(index_mcp.y * h))

                # Midpoint
                mid_point = ((thumb_mcp_coords[0] + index_mcp_coords[0]) / 2,
                             (thumb_mcp_coords[1] + index_mcp_coords[1]) / 2)
                mid_point = (int(mid_point[0]), int(mid_point[1]))

                # Calculate and send angle
                stable_angle = calculate_stable_angle_ema(mid_point, thumb_tip_coords, index_tip_coords)
                stable_angle = max(0, min(180, int(stable_angle)))  # clamp between 0-180
                send_servo_angle(stable_angle)

                # Draw visuals
                cv2.circle(frame, thumb_mcp_coords, 5, (0, 255, 0), -1)
                cv2.circle(frame, thumb_tip_coords, 5, (0, 0, 255), -1)
                cv2.circle(frame, index_tip_coords, 5, (255, 0, 0), -1)
                cv2.circle(frame, mid_point, 5, (255, 255, 255), -1)
                cv2.circle(frame, index_mcp_coords, 5, (0, 0, 0), -1)
                cv2.line(frame, thumb_mcp_coords, index_mcp_coords, (255, 100, 150), 2)
                cv2.line(frame, thumb_tip_coords, mid_point, (0, 255, 0), 2)
                cv2.line(frame, index_tip_coords, mid_point, (255, 0, 0), 2)
                cv2.putText(frame, f'Angle: {stable_angle} deg', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display current mode
    mode_text = "SAVE MODE: ON (Servo Active)" if save_mode else f"NORMAL MODE (Servo Hold {servo_angle}°)"
    cv2.putText(frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Detection", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        save_mode = True
        print("Save mode activated!")
        for finger in led_states.keys():
            led_states[finger] = (finger in fingers)
        send_led_state()  # send frozen LED state once
    elif key == ord('s'):
        save_mode = False
        print("Normal mode activated! Servo holding last angle.")

cap.release()
cv2.destroyAllWindows()
