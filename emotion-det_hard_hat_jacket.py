import cv2
import torch
from PIL import Image
from super_gradients.training import models
from deepface import DeepFace

# Set the device to GPU if available, otherwise fallback to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO-NAS model
yolo_model = models.get(
    model_name='yolo_nas_s',  # specify the model name here
    num_classes=2,
    checkpoint_path='yolo_nas/average_model.pth'
).to(DEVICE)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize frame counter
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Process emotion analysis on every 5th frame
    frame_count += 1
    if frame_count % 5 == 0:
        # Convert frame to PIL image for YOLO-NAS object detection
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform YOLO-NAS object detection
        yolo_results = yolo_model.predict([img], conf=0.75)
        yolo_boxes = yolo_results.prediction.bboxes_xyxy
        yolo_labels = yolo_results.prediction.labels
        yolo_scores = yolo_results.prediction.confidence
        class_names = ['Safety Helmet', 'Reflective Jacket'] 

        for box, label, score in zip(yolo_boxes, yolo_labels, yolo_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_names[label]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Combined Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
