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

# Define the ROI coordinates based on the frame size (848, 640)
roi_points = np.array([(350, 97), (541, 110), (645, 474), (279, 523)], np.int32)
roi_polygon = roi_points.reshape((-1, 1, 2))

# Define the target frame size (848x640)
target_width = 848
target_height = 640

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to the target size (848x640)
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Perform inference with the custom model
    results = model(frame_resized)
    person_masks = []

    for result in results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.data, result.boxes.cls):
                if cls.item() == 0:  # Check for "person" class (or your custom class ID)
                    person_masks.append(mask.cpu().numpy())

    # Get the frame dimensions (848x640 as specified)
    frame_height, frame_width = frame_resized.shape[:2]

    # Check if frame size is as expected
    print(f"Frame Size: {frame_width}x{frame_height}")  # Should print 848x640

    # Draw the ROI polygon on the resized frame (pink color)
    cv2.polylines(frame_resized, [roi_polygon], isClosed=True, color=(25, 105, 250), thickness=4)

    emergency_activated = False  # Flag to check if emergency is triggered

    for mask in person_masks:
        # Convert mask to binary (0 or 1)
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask_resized = cv2.resize(binary_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Create an overlay mask and apply color (red in this case)
        color_mask = np.zeros_like(frame_resized)
        color_mask[binary_mask_resized == 1] = [0, 0, 255]  # Red color
        frame_resized = cv2.addWeighted(frame_resized, 1, color_mask, 0.5, 0)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Draw contours on the frame
            cv2.drawContours(frame_resized, [contour], -1, (255, 0, 0), 2)  # Blue contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a bounding box around the detected object
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow bounding box

            # Check if the bounding box center is inside the ROI polygon
            center = (x + w // 2, y + h // 2)
            if cv2.pointPolygonTest(roi_polygon, center, False) >= 0:  # The point is inside the ROI
                emergency_activated = True
                # Print the emergency message in terminal
                print("Emergency Activated")
                # Show the emergency message on screen
                cv2.putText(frame_resized, "EMERGENCY ACTIVATED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

    # Resize frame for display (keep it at 848x640)
    cv2.imshow('Yolo Custom Class Detections', frame_resized)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()