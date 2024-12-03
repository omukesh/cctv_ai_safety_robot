import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# Load the YOLO segmentation model
model = YOLO('yolov8s-seg.pt')  # Update with the correct path to your model

load_dotenv()
# URL of the CCTV feed
cctv_url = os.getenv("CCTV_URL")
cap = cv2.VideoCapture(cctv_url)

# Define the target frame size
target_width = 848
target_height = 640

# Define circle parameters
circle_A_center = (340, 68)  # Point A
circle_A_radius = 25
circle_B_center = (210, 304)  # Point B
circle_B_radius = 40

# Define the ROI coordinates based on the frame size
roi_points = np.array([(350, 97), (541, 110), (645, 474), (279, 523)], np.int32)
roi_polygon = roi_points.reshape((-1, 1, 2))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to the target size (848x640)
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Draw yellow circles (constant)
    cv2.circle(frame_resized, circle_A_center, circle_A_radius, (0, 255, 255), 2)  # Yellow Circle A
    cv2.circle(frame_resized, circle_B_center, circle_B_radius, (0, 255, 255), 2)  # Yellow Circle B

    # Draw the ROI polygon
    cv2.polylines(frame_resized, [roi_polygon], isClosed=True, color=(25, 105, 250), thickness=4)

    # Extract regions inside the circles
    mask_A = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
    mask_B = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
    cv2.circle(mask_A, circle_A_center, circle_A_radius, 255, -1)
    cv2.circle(mask_B, circle_B_center, circle_B_radius, 255, -1)

    # Crop the circular regions from the frame
    region_A = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_A)
    region_B = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_B)

    # Convert regions to HSV for color detection
    hsv_A = cv2.cvtColor(region_A, cv2.COLOR_BGR2HSV)
    hsv_B = cv2.cvtColor(region_B, cv2.COLOR_BGR2HSV)

    # Define HSV range for green and red
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])
    red_lower_1 = np.array([0, 50, 50])
    red_upper_1 = np.array([10, 255, 255])
    red_lower_2 = np.array([170, 50, 50])
    red_upper_2 = np.array([180, 255, 255])

    # Check if green or red is present in region A
    green_A = cv2.inRange(hsv_A, green_lower, green_upper)
    red_A = cv2.inRange(hsv_A, red_lower_1, red_upper_1) | cv2.inRange(hsv_A, red_lower_2, red_upper_2)
    state_A = "Green" if np.sum(green_A) > np.sum(red_A) else "Red"

    # Check if green or red is present in region B
    green_B = cv2.inRange(hsv_B, green_lower, green_upper)
    red_B = cv2.inRange(hsv_B, red_lower_1, red_upper_1) | cv2.inRange(hsv_B, red_lower_2, red_upper_2)
    state_B = "Green" if np.sum(green_B) > np.sum(red_B) else "Red"

    print(f"Circle A color: {state_A}")
    print(f"Circle B color: {state_B}")

    # Perform inference with the YOLO segmentation model
    results = model(frame_resized)
    person_detected_in_roi = False

    for result in results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.data, result.boxes.cls):
                if cls.item() == 0:  # Check for "person" class
                    # Convert mask to binary (0 or 1)
                    binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                    binary_mask_resized = cv2.resize(binary_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

                    # Find contours in the binary mask
                    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        # Find the center of the bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        center = (x + w // 2, y + h // 2)

                        # Check if the center is inside the ROI
                        if cv2.pointPolygonTest(roi_polygon, center, False) >= 0:
                            person_detected_in_roi = True

                            # Apply the segmentation mask for visualization
                            color_mask = np.zeros_like(frame_resized)
                            color_mask[binary_mask_resized == 1] = [0, 0, 255]  # Red for person
                            frame_resized = cv2.addWeighted(frame_resized, 1, color_mask, 0.5, 0)

    # Check emergency condition
    if person_detected_in_roi and (state_A == "Green" or state_B == "Green"):
        cv2.putText(frame_resized, "EMERGENCY ACTIVATED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        print("Emergency Activated")
    elif person_detected_in_roi:
        cv2.putText(frame_resized, "SAFE TIME, NO ISSUES", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        print("Safe Time, No issues")

    # Display the frame
    cv2.imshow('Yolo Custom Class Detections', frame_resized)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
