import cv2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
cctv_url = os.getenv("CCTV_URL")

if not cctv_url:
    print("CCTV_URL not found in .env file")
    exit()

# Open the video stream
cap = cv2.VideoCapture(cctv_url)

if not cap.isOpened():
    print("Unable to open video stream. Check your RTSP URL and credentials.")
    exit()

print("Stream opened successfully. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Stream might be down.")
        break

    cv2.imshow("Stream Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
