import cv2
import time
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "best.onnx"
model = YOLO(MODEL_PATH, task="detect")

CONFIDENCE_THRESHOLD = 0.33
FOCAL_ESTIMATE = 500  # Distance calibration constant (pixels)
REAL_HUMAN_HEIGHT_M = 1.7  # Average human height in meters

cap = cv2.VideoCapture(0)
prev_time = 0
current_mode = "RGB"  # Modes: 'RGB', 'NIR', 'THERMAL'

print("🚀 TACTICAL TARGETING HUD ACTIVE")
print("Press '1' for RGB Mode | '2' for Simulated NIR | '3' for Simulated Thermal | 'q' to Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame, _ = frame.shape
    center_x, center_y = w_frame // 2, h_frame // 2

    # ==========================================
    # 2. MULTI-SPECTRAL SIMULATOR
    # ==========================================
    if current_mode == "NIR":
        # Grayscale with contrast boost to simulate NIR reflectivity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.equalizeHist(gray)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    elif current_mode == "THERMAL":
        # Heatmap colormap to simulate false-color LWIR Thermal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    else:
        processed_frame = frame.copy()

    # ==========================================
    # 3. MODEL INFERENCE & TARGETING
    # ==========================================
    results = model.predict(processed_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    primary_target = None
    max_area = 0

    # Draw center screen targeting crosshair
    cv2.line(processed_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
    cv2.line(processed_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Calculate bounding box metrics
        target_w = x2 - x1
        target_h = y2 - y1
        target_cx = x1 + (target_w // 2)
        target_cy = y1 + (target_h // 2)
        area = target_w * target_h

        # Track the largest target (closest human) as the primary target
        if area > max_area:
            max_area = area
            est_distance = round((REAL_HUMAN_HEIGHT_M * FOCAL_ESTIMATE) / max(target_h, 1), 1)
            primary_target = (target_cx, target_cy, x1, y1, x2, y2, est_distance, conf)

        # Draw secondary bounding box
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 165, 0), 1)

    # ==========================================
    # 4. AUTONOMOUS STEERING TELEMETRY
    # ==========================================
    steering_cmd = "SEARCHING..."
    cmd_color = (200, 200, 200)

    if primary_target:
        tcx, tcy, x1, y1, x2, y2, dist, conf = primary_target
        
        # Draw primary target lock frame & crosshair
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(processed_frame, (tcx, tcy), 6, (0, 0, 255), -1)
        
        # Vector line from screen center to target center
        cv2.line(processed_frame, (center_x, center_y), (tcx, tcy), (0, 255, 255), 2)

        # Calculate pixel offset error (Delta X)
        error_x = tcx - center_x
        deadzone = 50  # Pixels within center tolerance

        if dist < 1.0:
            steering_cmd = "🚨 PROXIMITY ALERT - STOP!"
            cmd_color = (0, 0, 255)
        elif error_x < -deadzone:
            steering_cmd = f"◄◄ STEER LEFT ({abs(error_x)}px)"
            cmd_color = (255, 255, 0)
        elif error_x > deadzone:
            steering_cmd = f"STEER RIGHT ►► ({abs(error_x)}px)"
            cmd_color = (255, 255, 0)
        else:
            steering_cmd = "LOCK ENGAGED - APPROACH"
            cmd_color = (0, 255, 0)

        # Target label
        cv2.putText(processed_frame, f"LOCKED | Dist: ~{dist}m | Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ==========================================
    # 5. HUD RENDERING
    # ==========================================
    # Calculate FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Top Telemetry Bar
    overlay = processed_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, processed_frame, 0.35, 0, processed_frame)

    cv2.putText(processed_frame, f"MODE: [{current_mode}]", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(processed_frame, f"FPS: {fps}", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(processed_frame, f"TARGETS: {len(results.boxes)}", (320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    # Steering Command Bar
    cv2.putText(processed_frame, f"NAV DIRECTIVE: {steering_cmd}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)

    cv2.imshow("Tactical Multi-Spectral Navigation HUD", processed_frame)

    # Keybind controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_mode = "RGB"
    elif key == ord('2'):
        current_mode = "NIR"
    elif key == ord('3'):
        current_mode = "THERMAL"

cap.release()
cv2.destroyAllWindows()