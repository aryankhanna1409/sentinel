import cv2
import time
from ultralytics import YOLO

# ==========================================
# 1. DASHBOARD CONFIGURATION
# ==========================================
# Load your optimized model (change to "best.pt" if you prefer)
MODEL_PATH = "best.onnx" 
model = YOLO(MODEL_PATH, task="detect")

# Set confidence to the mathematical peak from your F1 Curve
CONFIDENCE_THRESHOLD = 0.33 

# Open the default laptop webcam (Source 0)
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_frame_time = 0

print("🚀 Booting up Multi-Spectral Vision Dashboard... Press 'q' to close.")

# ==========================================
# 2. LIVE INFERENCE LOOP
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Waiting for camera feed...")
        continue

    # 1. Feed the frame to your trained model
    # verbose=False stops the terminal from spamming data every millisecond
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # 2. Ultralytics magic: instantly draw all bounding boxes and labels
    annotated_frame = results[0].plot()
    
    # 3. Extract telemetry data for the dashboard
    human_count = len(results[0].boxes)
    
    # 4. Calculate Live FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    # ==========================================
    # 3. DRAWING THE HUD DASHBOARD
    # ==========================================
    # Draw a semi-transparent black background box for readability
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
    # Apply the overlay with 60% opacity
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
    
    # Render Dashboard Text
    cv2.putText(annotated_frame, f"VISION SYSTEM ACTIVE", (25, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Live FPS: {int(fps)}", (25, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Targets Locked: {human_count}", (25, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # ==========================================
    # 4. DISPLAY THE FEED
    # ==========================================
    cv2.imshow("Multi-Spectral RC Vision", annotated_frame)
    
    # Press 'q' on your keyboard to safely exit the dashboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()