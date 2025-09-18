import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Load model and labels
model = load_model("sign_language_model.h5")
labels = ["Hello", "I love you", "Sorry", "Yes", "No"]

# Image preprocessing parameters
imgSize = 300
offset = 20

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize with aspect ratio handling
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prepare input for model
        imgInput = cv2.resize(imgWhite, (64, 64))
        imgInput = imgInput / 255.0  # Normalize pixel values
        imgInput = np.expand_dims(imgInput, axis=0)  # Add batch dimension

        # Predict with model
        predictions = model.predict(imgInput)
        index = np.argmax(predictions)
        predicted_label = labels[index]

        # Draw bigger green rectangle with label inside
        cv2.rectangle(img, (x - 10, y - 60), (x + w + 10, y - 20), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, predicted_label, (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10),
                      (0, 255, 0), thickness=5)

    # Display webcam feed
    cv2.imshow("Sign Language Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
